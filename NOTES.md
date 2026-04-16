18.02.2026: Notes on priors and training

Test iterative boltzmann inversion (might be too expensive, since would need to fold whole protein (or dataset?) in md simulation each iteration)

Repulsion needs to be much stronger! Currently maxes out at 0.2 kcal/mol (should be larger 2x or 3x of kT) 

Run training with SGD Nesterov and use as a baseline to compare other runs to

Test without weight decay and make sure that the weights do not decay during one epoch or before the model has seen enough data

Use a smaller model for testing, with larger batch sizes

Check for any performance bottlenecks (other than reducing model size or tuning other hyper-parameters)



Training Phase 1: 

General: Use small Allegro model; Run 40 epochs; All (?) need to be resolved before run can be started;

Round 1: Allegro only; SGD Nesterov; Stronger repulsion; No weight decay
- Implementation needed: get SGD with nesterov setting turned on from optax 

Round 2: Allegro + priors (splines); SGD Nesterov; Stronger repulsion; no weight decay

Round 3: Prior only; Run 5 epochs; 
- Spline priors
- "Normal" priors
- Pretrained priors (what parameters are we actually updating?)
- Train priors (what parameters are we actually updating?)

Round 4: Allegro + priors (splines); Adabelief; Strong repulsion
- Higher beta values
- Weight decay (depending on rounds 1 and 2?)

Round 5: tbd 
- Use knowledge/strange behavior from previous rounds


Optimization changes: 

Need to profile 
Check for any recompilation

Maybe remove numpy data loader and use a jax variant. 

17.03.2026
Reduce weight of evaluate convergence (post training chemtrain module). Might not need early stopping or per-epoch validation loss, or can trim it some other way

Still need to fix the export of the model. Rather need to recompile my LAMMPS connector due to a changed cuda version


19.03.2026
Standard deviation of forces. Trained vs. 0 prediction. 
Safe and minimal LJ prior for anti MD explosions.


31.03.2026
There is an issue with exporting of the uniform1d backend. Currently model is written into naive tp method after training. Need to check this out!




15.04.2026
Continue working on rigurous md testing. 
Do the constrained md test (option3)
Get original gromacs config for actual comparisons
Talk to emile about tica analysis and set it up correctly. 
Once MD testing results are in, look for: 
	Noise to Signal ratio
	How well we approx grad PMF
	Sample transition states? 
	Do we need priors for long MD runs? 
Maybe test specific, known protein systems and see if I can recreate them using a trained model. 
Test out of distribution!
After results are evaluated, think about next steps. 


