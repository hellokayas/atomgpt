### Downloading the models.

Running the code in /projects/p32726/microscopy-gpt/atomgpt/atomgpt/models/download_models.py should download the models and keep them inside /projects/p32726/microscopy-gpt/atomgpt/atomgpt/models/unsloth. These would be the base models. To download them, you will need your own HuggingFace token.

### The trained checkpoints

1. Models trained on DFT-2d: https://drive.google.com/drive/folders/1AiaShQvTmDrhXQSfWUYj_FAEcMINBt-F?usp=sharing
2. Models trained on c2db: https://drive.google.com/drive/folders/1awOFP4KkEHzTIcxf_Xl84DDkAhOKINhJ?usp=sharing

Inside each of these folders, you will also find the csv files that were generated when the models were run on their respective test datasets.

### Now running the following code will let you play around with whatever images you want and also the prompt and a couple of other params like top_p and temp.

/projects/p32726/microscopy-gpt/atomgpt/atomgpt/inverse_models/custom_test.py

You might need to set the image paths, etc., as you choose, or you need to keep your own images in the folder mentioned (/projects/p32726/microscopy-gpt/atomgpt/atomgpt/formula_based)

