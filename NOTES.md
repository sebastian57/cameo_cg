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


