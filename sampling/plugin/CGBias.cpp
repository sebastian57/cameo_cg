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

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/un.h>
#include <unistd.h>

// Optional in-process backend (see the CGBIAS_WITH_CONNECTOR notes further down). This MUST
// be included at global scope: inside namespace PLMD the library's `jcn` namespace would be
// nested and nothing would link.
#ifdef CGBIAS_WITH_CONNECTOR
#include "libconnector.h"
#endif

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

// ---------------------------------------------------------------------------
// Optional in-process backend: evaluate an exported model through chemtrain-deploy's
// libconnector instead of shipping positions over the socket. Compile with
// -DCGBIAS_WITH_CONNECTOR and link -lconnector; without it the plugin builds exactly as
// before and MODEL= is rejected at parse time.
//
// UNITS. PLUMED is internally nm / kJ-mol regardless of any UNITS line in the input (UNITS
// only affects parsing and printing). The exported models are "real" units: Angstrom and
// kcal/mol. So positions are scaled by 10 on the way in, and the returned energy and forces
// by 4.184 and 41.84 on the way out -- the same constants sampling/protocol.py applies on
// the Python route.
//
// SPECIES. The connector subtracts one from every type internally, so types are passed
// ONE-BASED here. Passing model-convention (zero-based) values would silently shift every
// species by one and produce a plausible wrong energy. See export_check/MD_DEPLOY_CORR.md.
// ---------------------------------------------------------------------------
// NOTE: libconnector.h is included at the TOP of this file, outside namespace PLMD.
// Including it here would nest the library's namespace as PLMD::colvar::jcn and every
// symbol would fail to resolve at dlopen time with an undefined-symbol error naming
// _ZN4PLMD6colvar3jcn... instead of _ZN3jcn...

namespace {
constexpr double A_PER_NM = 10.0;
constexpr double KJ_PER_KCAL = 4.184;
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

  // ---- in-process compiled-model backend ----
  std::string modelPath_;
  bool useConnector_;
  std::vector<int> species_;      // one-based, as the connector expects
  double cutoff_;
  bool halfList_;
  // Rank decomposition over PLUMED's intra-replica communicator. PLUMED shares the
  // requested atoms to every rank, so each rank holds all beads but OWNS a contiguous
  // slice; ghosts are the remaining beads within cutoff. Forces are reduced with comm.Sum.
  std::size_t ownBegin_, ownEnd_;
#ifdef CGBIAS_WITH_CONNECTOR
  std::unique_ptr<jcn::Connector> connector_;
#endif
  void evaluateConnector();

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
  keys.add("optional", "SOCKET", "Unix-domain socket served by sampling/server.py. "
           "Mutually exclusive with MODEL.");
  keys.add("optional", "MODEL",
           "exported model (compiled MLIR) evaluated in-process through libconnector, with "
           "no Python in the timestep loop. Mutually exclusive with SOCKET.");
  keys.add("optional", "SPECIES",
           "per-bead species indices in MODEL convention (zero-based); one is added before "
           "they reach the connector. Defaults to 0,1,...,n-1.");
  keys.add("optional", "BACKEND", "connector backend, 'cpu' (default) or a PJRT plugin name");
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
  parse("MODEL", modelPath_);
  if(socketPath_.empty() == modelPath_.empty()) {
    error("CG_BIAS needs exactly one of SOCKET= (Python bias server) or MODEL= "
          "(compiled model via libconnector)");
  }
  useConnector_ = !modelPath_.empty();
  parse("RECOMPUTE_STRIDE", recomputeStride_);
  if(recomputeStride_ < 1) error("RECOMPUTE_STRIDE must be >= 1");
  parse("TIMEOUT", timeout_);
  if(timeout_ <= 0.0) error("TIMEOUT must be > 0");
  if(!socketPath_.empty() && socketPath_.size() >= sizeof(sockaddr_un::sun_path)) {
    error("SOCKET path is too long");
  }
  std::vector<int> speciesIn;
  parseVector("SPECIES", speciesIn);
  std::string backend = "cpu";
  parse("BACKEND", backend);
  checkRead();

  species_.resize(nAtoms_);
  for(std::size_t i = 0; i < nAtoms_; ++i) {
    // +1: the connector subtracts one internally.
    species_[i] = (speciesIn.empty() ? static_cast<int>(i) : speciesIn.at(i)) + 1;
  }
  if(!speciesIn.empty() && speciesIn.size() != nAtoms_) {
    error("SPECIES must have one entry per atom in ATOMS");
  }

  // Contiguous ownership slice for this rank; ghosts are handled in evaluateConnector().
  {
    const std::size_t nRanks = std::max<std::size_t>(1, comm.Get_size());
    const std::size_t rank = comm.Get_rank();
    const std::size_t base = nAtoms_ / nRanks, rem = nAtoms_ % nRanks;
    ownBegin_ = rank * base + std::min<std::size_t>(rank, rem);
    ownEnd_ = ownBegin_ + base + (rank < rem ? 1 : 0);
  }

  // header (4 x 8 bytes) + payload
  cachedForces_.assign(3 * nAtoms_, 0.0);
  requestBuffer_.assign(4 * sizeof(std::uint64_t) + 3 * nAtoms_ * sizeof(double), 0);
  responseBuffer_.assign(
      4 * sizeof(std::uint64_t) + sizeof(double) + 3 * nAtoms_ * sizeof(double), 0);

  addValueWithDerivatives();
  setNotPeriodic();
  requestAtoms(atoms);

  if(useConnector_) {
#ifndef CGBIAS_WITH_CONNECTOR
    error("CG_BIAS MODEL= requires the plugin to be built with -DCGBIAS_WITH_CONNECTOR "
          "and linked against libconnector; rebuild with sampling/plugin/compile_plugin.sh "
          "--with-connector");
#else
    std::ifstream blob(modelPath_, std::ios::binary);
    if(!blob) error("Could not open CG_BIAS MODEL file " + modelPath_);
    const std::string modelBytes((std::istreambuf_iterator<char>(blob)),
                                 std::istreambuf_iterator<char>());
    jcn::ConnectorConfig ccfg;
    ccfg.backend = backend;
    ccfg.device = 0;
    ccfg.memory_fraction = 0.25f;   // several ranks may share one GPU
    connector_ = std::make_unique<jcn::Connector>(ccfg);
    jcn::ModelConfig mcfg;
    mcfg.model = modelBytes;
    mcfg.newton = true;
    const jcn::ModelProperties props = connector_->load_model(mcfg);
    cutoff_ = props.cutoff;
    halfList_ = props.neighbor_list.half_list;
    log.printf("  CG_BIAS model  : %s (backend %s)\n", modelPath_.c_str(), backend.c_str());
    log.printf("  model cutoff   : %g A, half_list %d, unit_style %s\n",
               cutoff_, static_cast<int>(halfList_), props.unit_style);
    log.printf("  rank %u/%u owns beads [%u,%u)\n",
               static_cast<unsigned>(comm.Get_rank()), static_cast<unsigned>(comm.Get_size()),
               static_cast<unsigned>(ownBegin_), static_cast<unsigned>(ownEnd_));
#endif
    return;   // no socket in this mode
  }

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

// Evaluate the compiled model in-process, distributed over PLUMED's intra-replica
// communicator. Fills cachedEnergy_ / cachedForces_ in PLUMED units.
//
// PLUMED shares the requested atoms to every rank, so each rank already holds all bead
// positions. It therefore does not need a halo exchange -- it just picks its owned slice,
// gathers the beads within cutoff as ghosts, and evaluates that subdomain. The connector's
// (n_local, n_ghost) interface is exactly this decomposition, and it returns ghost forces
// under Newton's third law, so summing every rank's contribution over comm gives the total.
void CGBias::evaluateConnector() {
#ifdef CGBIAS_WITH_CONNECTOR
  std::fill(cachedForces_.begin(), cachedForces_.end(), 0.0);
  cachedEnergy_ = 0.0;

  const std::size_t nOwn = ownEnd_ - ownBegin_;
  if(nOwn > 0) {
    // positions in Angstrom, owned beads first then ghosts (the layout the connector wants)
    std::vector<std::size_t> globalIndex;
    globalIndex.reserve(nAtoms_);
    for(std::size_t i = ownBegin_; i < ownEnd_; ++i) globalIndex.push_back(i);
    const double cut2 = cutoff_ * cutoff_;
    for(std::size_t j = 0; j < nAtoms_; ++j) {
      if(j >= ownBegin_ && j < ownEnd_) continue;
      const Vector pj = getPosition(static_cast<unsigned>(j));
      bool near = false;
      for(std::size_t i = ownBegin_; i < ownEnd_ && !near; ++i) {
        const Vector d = pj - getPosition(static_cast<unsigned>(i));
        near = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) * A_PER_NM * A_PER_NM <= cut2;
      }
      if(near) globalIndex.push_back(j);
    }
    const std::size_t nTotal = globalIndex.size();
    const std::size_t nGhost = nTotal - nOwn;

    std::vector<std::vector<double>> xs(nTotal, std::vector<double>(3, 0.0));
    std::vector<std::vector<double>> fs(nTotal, std::vector<double>(3, 0.0));
    std::vector<double*> xp(nTotal), fp(nTotal);
    std::vector<int> types(nTotal);
    for(std::size_t k = 0; k < nTotal; ++k) {
      const Vector p = getPosition(static_cast<unsigned>(globalIndex[k]));
      for(unsigned d = 0; d < 3; ++d) xs[k][d] = p[d] * A_PER_NM;
      xp[k] = xs[k].data();
      fp[k] = fs[k].data();
      types[k] = species_[globalIndex[k]];
    }

    // LAMMPS-style CSR neighbour list, brute force. Trivial at CG bead counts, and the
    // indices must be local to this subdomain, not global.
    std::vector<int> ilist(nOwn), numneigh(nOwn, 0);
    std::vector<std::vector<int>> storage(nOwn);
    std::vector<int*> firstneigh(nOwn);
    for(std::size_t i = 0; i < nOwn; ++i) {
      ilist[i] = static_cast<int>(i);
      for(std::size_t j = 0; j < nTotal; ++j) {
        if(j == i) continue;
        if(halfList_ && j < i) continue;   // canonical (min,max) ordering
        const double dx = xs[i][0] - xs[j][0], dy = xs[i][1] - xs[j][1],
                     dz = xs[i][2] - xs[j][2];
        if(dx * dx + dy * dy + dz * dz <= cut2) storage[i].push_back(static_cast<int>(j));
      }
      numneigh[i] = static_cast<int>(storage[i].size());
      firstneigh[i] = storage[i].empty() ? nullptr : storage[i].data();
    }

    jcn::Results results;
    try {
      results = connector_->compute_force(
          static_cast<int>(nOwn), static_cast<int>(nGhost), xp.data(), fp.data(),
          types.data(), static_cast<int>(nOwn), ilist.data(), numneigh.data(),
          firstneigh.data(), /*list_changed=*/true, /*allow_recompile=*/false);
    } catch(const jcn::RecompilationRequired&) {
      // Buffer capacity grew; retry once with compilation permitted. Settles after the
      // first step at fixed system size.
      results = connector_->compute_force(
          static_cast<int>(nOwn), static_cast<int>(nGhost), xp.data(), fp.data(),
          types.data(), static_cast<int>(nOwn), ilist.data(), numneigh.data(),
          firstneigh.data(), true, true);
    }

    cachedEnergy_ = results.potential * KJ_PER_KCAL;
    // forces are kcal/mol/A -> kJ/mol/nm, scattered back to global bead indices
    for(std::size_t k = 0; k < nTotal; ++k) {
      const std::size_t g = globalIndex[k];
      for(unsigned d = 0; d < 3; ++d) {
        cachedForces_[3 * g + d] += fs[k][d] * KJ_PER_KCAL * A_PER_NM;
      }
    }
  }

  // Every rank contributed its own subdomain; sum to get the whole bias.
  if(comm.Get_size() > 1) {
    comm.Sum(cachedEnergy_);
    comm.Sum(cachedForces_);
  }
  haveCached_ = true;
#endif
}

void CGBias::calculate() {
  const long long mdStep = static_cast<long long>(getStep());
  const bool recompute =
      (!haveCached_) || (recomputeStride_ <= 1) || (mdStep % recomputeStride_ == 0);

  if(useConnector_) {
    if(recompute) evaluateConnector();
    setValue(cachedEnergy_);
    for(std::size_t atom = 0; atom < nAtoms_; ++atom) {
      setAtomsDerivatives(static_cast<unsigned>(atom),
                          Vector(-cachedForces_[3 * atom + 0],
                                 -cachedForces_[3 * atom + 1],
                                 -cachedForces_[3 * atom + 2]));
    }
    Tensor boxDerivative;
    boxDerivative.zero();
    setBoxDerivatives(boxDerivative);
    return;
  }

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
