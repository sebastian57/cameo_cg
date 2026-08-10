// PLUMED action for CG-guided atomistic enhanced sampling.
//
// Pure transport: sends the mapped CG bead positions to a Unix-domain socket and
// applies the returned energy and forces. It carries NO bias logic of its own --
// alpha schedules, TICA targets, term composition and diagnostics all live in the
// Python server (sampling/server.py + sampling/biases/). That separation is why the
// same compiled plugin serves any combination of biases without a rebuild.
//
// Clean break from the v1 MLCG_TEACHER action, which hardcoded five atoms into
// fixed-size 15-double structs on both ends. Here the atom count is whatever ATOMS=
// declares, and it travels in the wire header so the server can validate it.
//
// Wire format (native byte order, same node; see sampling/protocol.py):
//   request  : magic(u64) version(u64) step(i64) n_atoms(i64) pos_nm[3n](f64)
//   response : magic(u64) version(u64) step(i64) n_atoms(i64) energy_kj(f64)
//              forces_kj_nm[3n](f64)
//
// Build: sampling/plugin/compile_plugin.sh

#include "plumed/core/Colvar.h"
#include "plumed/core/ActionRegister.h"

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/un.h>
#include <unistd.h>

namespace PLMD {
namespace colvar {

namespace {

constexpr std::uint64_t REQUEST_MAGIC = 0x4347425245513200ULL;   // CGBREQ2
constexpr std::uint64_t RESPONSE_MAGIC = 0x4347425245533200ULL;  // CGBRES2
constexpr std::uint64_t PROTOCOL_VERSION = 2ULL;

// Both directions carry SO_SNDTIMEO/SO_RCVTIMEO (set in registerKeywords/connect).
// Without them a dead or wedged bias server leaves GROMACS blocked in read()
// forever, holding the SLURM allocation to walltime with nothing in any log. The
// errors below name the socket and step so the failing replica is identifiable.
void writeAll(const int fd, const void* buffer, std::size_t size,
              const std::string& where) {
  const char* ptr = static_cast<const char*>(buffer);
  while(size > 0) {
    const ssize_t count = ::write(fd, ptr, size);
    if(count < 0 && errno == EINTR) continue;
    if(count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
      throw std::runtime_error("CG_BIAS socket write TIMED OUT (" + where +
                               "); bias server is not consuming requests");
    if(count <= 0) throw std::runtime_error("CG_BIAS socket write failed (" + where + ")");
    ptr += count;
    size -= static_cast<std::size_t>(count);
  }
}

void readAll(const int fd, void* buffer, std::size_t size, const std::string& where) {
  char* ptr = static_cast<char*>(buffer);
  while(size > 0) {
    const ssize_t count = ::read(fd, ptr, size);
    if(count < 0 && errno == EINTR) continue;
    if(count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
      throw std::runtime_error("CG_BIAS socket read TIMED OUT (" + where +
                               "); bias server stalled or died");
    if(count <= 0) throw std::runtime_error("CG_BIAS socket read failed (" + where + ")");
    ptr += count;
    size -= static_cast<std::size_t>(count);
  }
}

// Apply a send/receive deadline to the connected socket.
void setSocketTimeout(const int fd, const double seconds) {
  struct timeval tv;
  tv.tv_sec = static_cast<time_t>(seconds);
  tv.tv_usec = static_cast<suseconds_t>((seconds - static_cast<double>(tv.tv_sec)) * 1e6);
  ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
}

}  // namespace

class CGBias : public Colvar {
  std::string socketPath_;
  std::size_t nAtoms_;
  int socketFd_;
  std::vector<char> requestBuffer_;
  std::vector<char> responseBuffer_;
  // Recompute stride. The bias is APPLIED every step; only the socket round-trip is
  // strided, with the previous energy/forces held constant in between. Striding the
  // application instead would cut the time-averaged bias by the same factor, which is
  // a different simulation rather than a cheaper one.
  long long recomputeStride_;
  // Per-I/O deadline in seconds; without it a wedged server blocks GROMACS forever.
  double timeout_;
  bool haveCached_;
  double cachedEnergy_;
  std::vector<double> cachedForces_;

public:
  explicit CGBias(const ActionOptions&);
  ~CGBias() override {
    if(socketFd_ >= 0) ::close(socketFd_);
  }
  void calculate() override;
  static void registerKeywords(Keywords&);
};

PLUMED_REGISTER_ACTION(CGBias, "CG_BIAS")

void CGBias::registerKeywords(Keywords& keys) {
  Colvar::registerKeywords(keys);
  keys.add("atoms", "ATOMS", "the CG bead atoms, in mapping order; any count");
  keys.add("compulsory", "SOCKET", "Unix-domain socket served by sampling/server.py");
  keys.add("optional", "RECOMPUTE_STRIDE",
           "recompute the bias every N steps and hold it constant in between "
           "(default 1). The bias is applied on every step regardless.");
  keys.add("optional", "TIMEOUT",
           "seconds to wait on any single socket read/write before aborting "
           "(default 300). Prevents a dead bias server from holding the allocation.");
}

CGBias::CGBias(const ActionOptions& options)
    : PLUMED_COLVAR_INIT(options), nAtoms_(0), socketFd_(-1),
      recomputeStride_(1), timeout_(300.0), haveCached_(false), cachedEnergy_(0.0) {
  std::vector<AtomNumber> atoms;
  parseAtomList("ATOMS", atoms);
  if(atoms.empty()) error("CG_BIAS requires at least one atom in ATOMS");
  nAtoms_ = atoms.size();
  parse("SOCKET", socketPath_);
  parse("RECOMPUTE_STRIDE", recomputeStride_);
  if(recomputeStride_ < 1) error("RECOMPUTE_STRIDE must be >= 1");
  parse("TIMEOUT", timeout_);
  if(timeout_ <= 0.0) error("TIMEOUT must be > 0");
  if(socketPath_.size() >= sizeof(sockaddr_un::sun_path)) error("SOCKET path is too long");
  checkRead();

  // header (4 x 8 bytes) + payload
  cachedForces_.assign(3 * nAtoms_, 0.0);
  requestBuffer_.assign(4 * sizeof(std::uint64_t) + 3 * nAtoms_ * sizeof(double), 0);
  responseBuffer_.assign(
      4 * sizeof(std::uint64_t) + sizeof(double) + 3 * nAtoms_ * sizeof(double), 0);

  addValueWithDerivatives();
  setNotPeriodic();
  requestAtoms(atoms);

  socketFd_ = ::socket(AF_UNIX, SOCK_STREAM, 0);
  if(socketFd_ < 0) error("Could not create CG_BIAS socket");
  sockaddr_un address;
  std::memset(&address, 0, sizeof(address));
  address.sun_family = AF_UNIX;
  std::strncpy(address.sun_path, socketPath_.c_str(), sizeof(address.sun_path) - 1);
  if(::connect(socketFd_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
    error("Could not connect to CG_BIAS socket " + socketPath_ +
          " (is sampling/server.py running for this replica?)");
  }
  setSocketTimeout(socketFd_, timeout_);
  log.printf("  CG_BIAS socket : %s\n", socketPath_.c_str());
  log.printf("  socket timeout : %g s\n", timeout_);
  log.printf("  CG beads       : %u (protocol v%llu)\n",
             static_cast<unsigned>(nAtoms_),
             static_cast<unsigned long long>(PROTOCOL_VERSION));
  log.printf("  all bias logic (alpha schedules, TICA, term composition) lives server-side\n");
}

void CGBias::calculate() {
  const long long mdStep = static_cast<long long>(getStep());
  const bool recompute =
      (!haveCached_) || (recomputeStride_ <= 1) || (mdStep % recomputeStride_ == 0);

  if(recompute) {
  char* out = requestBuffer_.data();
  const std::uint64_t reqMagic = REQUEST_MAGIC;
  const std::uint64_t version = PROTOCOL_VERSION;
  const std::int64_t step = static_cast<std::int64_t>(getStep());
  const std::int64_t nAtoms = static_cast<std::int64_t>(nAtoms_);
  std::memcpy(out + 0 * sizeof(std::uint64_t), &reqMagic, sizeof(reqMagic));
  std::memcpy(out + 1 * sizeof(std::uint64_t), &version, sizeof(version));
  std::memcpy(out + 2 * sizeof(std::uint64_t), &step, sizeof(step));
  std::memcpy(out + 3 * sizeof(std::uint64_t), &nAtoms, sizeof(nAtoms));
  double* pos = reinterpret_cast<double*>(out + 4 * sizeof(std::uint64_t));
  for(std::size_t atom = 0; atom < nAtoms_; ++atom) {
    const Vector position = getPosition(static_cast<unsigned>(atom));
    for(unsigned dim = 0; dim < 3; ++dim) pos[3 * atom + dim] = position[dim];
  }
  writeAll(socketFd_, requestBuffer_.data(), requestBuffer_.size(),
           socketPath_ + " step " + std::to_string(mdStep));

  readAll(socketFd_, responseBuffer_.data(), responseBuffer_.size(),
          socketPath_ + " step " + std::to_string(mdStep));
  const char* in = responseBuffer_.data();
  std::uint64_t respMagic = 0, respVersion = 0;
  std::int64_t respStep = 0, respAtoms = 0;
  std::memcpy(&respMagic, in + 0 * sizeof(std::uint64_t), sizeof(respMagic));
  std::memcpy(&respVersion, in + 1 * sizeof(std::uint64_t), sizeof(respVersion));
  std::memcpy(&respStep, in + 2 * sizeof(std::uint64_t), sizeof(respStep));
  std::memcpy(&respAtoms, in + 3 * sizeof(std::uint64_t), sizeof(respAtoms));
  if(respMagic != RESPONSE_MAGIC) error("CG_BIAS: bad response magic from server");
  if(respVersion != PROTOCOL_VERSION) {
    error("CG_BIAS: protocol version mismatch; rebuild the plugin or update the server");
  }
  if(respAtoms != nAtoms) error("CG_BIAS: server returned the wrong atom count");

  double energy = 0.0;
  std::memcpy(&energy, in + 4 * sizeof(std::uint64_t), sizeof(energy));
  const double* forces =
      reinterpret_cast<const double*>(in + 4 * sizeof(std::uint64_t) + sizeof(double));
  cachedEnergy_ = energy;
  for(std::size_t k = 0; k < 3 * nAtoms_; ++k) cachedForces_[k] = forces[k];
  haveCached_ = true;
  }  // end recompute

  // Applied EVERY step, from the cache when this step did not recompute.
  // The server returns the bias energy and the FORCES it implies. PLUMED wants the
  // value and its derivatives, and derivative = -force.
  setValue(cachedEnergy_);
  for(std::size_t atom = 0; atom < nAtoms_; ++atom) {
    const Vector derivative(-cachedForces_[3 * atom + 0],
                            -cachedForces_[3 * atom + 1],
                            -cachedForces_[3 * atom + 2]);
    setAtomsDerivatives(static_cast<unsigned>(atom), derivative);
  }
  Tensor boxDerivative;
  boxDerivative.zero();
  setBoxDerivatives(boxDerivative);
}

}  // namespace colvar
}  // namespace PLMD
