# Evaluating a model performance for CIFAR10-C (PyTorch)

## CIFAR10-C Examples
![cifar10-c-examples](./cifar10-c-examples.png)

## Preparation
### Download data
Download CIFAR10-C dataset from [here](https://zenodo.org/record/2535967#.XncuG5P7TUJ).  

### (Optional) Set your model
A default model is ResNet56. it's model file is already contained in `src/models/`.  
If you use your custom model, please do the following procedure.
- Add your model file to `src/models/`. 
- Add a new model option to `test.py`.

## Evaluation
```
# ResNet56
python src/test.py --weight_path <path/to/weight/file> --data_root <path/to/parent/directory/of/cifar10-c>

# Your custom model
python src/test.py --arch <your/model/name> --weight_path <path/to/weight/file> --data_root <path/to/parent/directory/of/cifar10-c>
```

### Output 
All corruption accuracy.  
This figure will be saved in `figs/`.

<img src=./naturally_trained.png width=500>

### (Optional) Other Useful Options
- `fig_dir` : Specify the directory path to save output figure.
- `weight_dir` : Execute evaluation for all weights files in the specified directory.
- `corruptions` : Evaluate a model performance for the specified corruption names.