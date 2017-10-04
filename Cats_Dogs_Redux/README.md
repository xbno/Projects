# Cats and Dogs Redux Competition on Kaggle

This project was inspired by the Fast-Ai course. I've modified th notebooks to suit my needs which includes all notebooks that helped me score a .08891 on the public leaderboard. This was my first try at using Convnets and I'm happy with the results yet there is much to learn!

In addition, I upgraded my home workstation from a windows10 pc to ubuntu 16.04. This took a bit of work but overall gives me far more freedom in terms of package requirements and now I can ssh and run remote jupyter notebooks from anywhere I am.

My machine:
  - Ubuntu, GTX-1060 6gb, 16gb ram

Notebooks:
  - 01-sample
    - Creates a sample set based on the data directory structure shown below
  - 02-prototype
    - My attempts at a few architectures. All are finetuned version of VGG16
  - 03-ensemble
    - Choose a selected prototype and create 3 models trained on the full dataset. This caused difficulty for my machine as the kernel will fail when multiple models are constructed/run.
  - 04-finetune
    - I also tried my hand at creating a single model which performed better than the ensemble I created. This should be rolled back into the ensemble technique to get a better score.
