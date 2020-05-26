# language-comparison-demo
This repo shows a simple comparison between python and Julia implementing logistic regression from scratch (no regularisation).

### How to use
1) Run create_model_data.py to generate some fake data to build the model on. 
This will be saved to a csv file 'fake_data.csv'. 
Please be aware I was using Spyder and had a project set up, so the path is relevant to the project, however you may need to amend the file path if not using similar.

2) You can now run either the Julia (`grad_descent.jl`) or python script (`grad_descent_example.py`). 
You may need to fiddle with the filepath for reading in the data depending on your setup (definitely for the Julia script as I haven't got relative pathing working).
