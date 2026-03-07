VAE — The Big Picture First                                                                                                                                                                                        
                                                                                                                                                                                                                     
  Regular Autoencoder:                                                                                                                                                                                               
    image → encoder → single point in latent space → decoder → reconstructed image              
    Problem: latent space has "holes" — can't generate new images                                                                                                                                                    
                                                                                                                                                                                                                     
  VAE:
    image → encoder → DISTRIBUTION (mean + variance) → sample → decoder → reconstructed image
    Magic: smooth latent space — sample ANYWHERE and get a valid image!


    ┌─────────┐      ┌─────────┐      ┌─────────┐       ┌─────────┐
    │  Image  │─────▶│ Encoder │─────▶│ Sample  │──────▶│ Decoder │──▶ Reconstructed
    │ (784)   │      │  MLP    │      │(reparam)│       │  MLP    │    Image (784)
    └─────────┘      └────┬────┘      └─────────┘       └─────────┘
                          │
                     ┌────┴────┐
                     │  μ      │  ← "center of the cloud"
                     │  log σ² │  ← "spread of the cloud"
                     └─────────┘

  Task 1 — The Encoder

  The encoder takes an image and outputs WHERE it lives in latent space — but as a cloud, not a point.

  Input: flattened MNIST image (784 floats)
      │
      ▼
  Linear(784, 256) + ReLU      ← compress
      │
      ├──▶ Linear(256, 2) → μ       ← "center"    (latent_dim=2)
      │
      └──▶ Linear(256, 2) → log σ²  ← "spread"    (latent_dim=2)

  Why two outputs?
  Regular autoencoder:  image → [0.5, -0.3]           ← one fixed point
  VAE encoder:          image → μ=[0.5, -0.3]         ← center of a cloud
                                log σ²=[-1.2, -0.8]   ← how wide the cloud is

  Why log σ² instead of σ²?
  σ² must be positive (can't have negative variance)
  log σ² can be any number — easier for the network to output
  When you need σ², just do exp(log σ²) — always positive!
