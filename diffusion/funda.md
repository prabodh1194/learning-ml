The core idea of diffusion is stupidly simple:                                                                                                                                                                     
                                                                                                                                                                                                                 
TRAINING:   take a real image, add noise, train a network to PREDICT the noise                                                                                                                                     
GENERATING: start with pure noise, remove noise step by step → image appears                                                                                                                                       

It's like:
  Training:    take a clean room, mess it up, teach a robot to clean
  Generating:  give the robot a messy room → it cleans it up!

The forward process is the "messing up" part. You destroy an image in 1000 tiny steps:

t=0          t=200        t=500        t=800        t=999
┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐
│  cat │    │ fuzzy│    │ blob │    │static│    │ pure │
│  pic │    │  cat │    │      │    │      │    │noise │
└──────┘    └──────┘    └──────┘    └──────┘    └──────┘
  clean      a bit       half        mostly      100%
             noisy       gone        noise       noise

The shortcut formula — you can jump to ANY step directly:

x_t = √(α̅_t) * x_0  +  √(1 - α̅_t) * ε
         ↑                 ↑        ↑
"how much of the     "how much    random
 original to keep"    noise to     noise
                      add"

When t=0:    α̅_0 ≈ 1.0  → x_t ≈ x_0           (all original, no noise)
When t=999:  α̅_999 ≈ 0.0 → x_t ≈ ε            (no original, all noise)

Don't worry about how α̅_t is computed yet. For now, just know it goes from ~1.0 to ~0.0.
