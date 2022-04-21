<!-- Project Name -->
# ![openai-gym-neat-games-ai](https://user-images.githubusercontent.com/95453430/164316726-ccdf6cd1-9477-4fa5-8a4f-46bb593f051e.svg)

<!-- Project Images -->

![Main Project Image (2)](https://user-images.githubusercontent.com/95453430/164322954-8513aab6-77b2-4de0-95e0-83e1bac98302.png)

<!-- Project Description -->
# ![project-description (15)](https://user-images.githubusercontent.com/95453430/164316736-e7d901d6-8fbc-4fca-9d62-2d3124986ff5.svg)

This is a **Python Project Repository** in which we use **Neuro Evolution of Augmented Topologies (NEAT) Algorithm to train and test diffrent neural networks to play 4 games provided by OpenAI Gym**. These four games are **Bipedal Walker-v3, Cart Pole-v1, Mountain Car-v0, & Lunar Lander-v2** as seen in the preview image above. In this repository, each game has a **script for training the neural network, a script for testing the neural network on the game, & a NEAT Configuration File** located in the **TrainingScripts, TestingScripts, & ConfigFiles** respectively. After a model is trained for a game, The **Neural Network** with the best **Fitness** is stored as pickle file in the **Winners** folder. This repository already comes with trained models for each of the games, so feel free to test the models in the games using the testing scripts!

<!-- Project Tech-Stack -->
# ![technologies-used (15)](https://user-images.githubusercontent.com/95453430/164316742-1e6675db-1d60-42f2-b566-abed44c9aae8.svg)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![OpenAI Gym](https://img.shields.io/badge/OpenAI%20Gym-0081A5?style=for-the-badge&logo=OpenAI-Gym&logoColor=white)
![NEAT Python](https://img.shields.io/badge/NEAT%20python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Figma](https://img.shields.io/badge/figma-%23F24E1E.svg?style=for-the-badge&logo=figma&logoColor=white)

<!-- How To Use Project -->
# ![how-to-use-project (10)](https://user-images.githubusercontent.com/95453430/164316747-f096ecad-cdad-40dd-8184-6dd486ec46c7.svg)

**Install the following Python libraries in your Virtual Environment using PIP**.

*Note: The library names are **CASE-SENSITIVE** for PIP installations below. Make sure your type them correctly.*

*Install NEAT for Python*
```Python
pip install neat-python
```

*Install OpenAI Gym for Python*
```Python
pip install gym
```

*Install Numpy for Python*
```Python
pip install numpy
```

Download a copy of this repository onto your local machine and extract it into a suitable folder.
- Open an IDE (VSCode Recommended) in the **Root Directory** of the Project. Make sure you follow this step for the scripts to work.
- Create a Virtual Environment in that folder.
- Install all the required Python libraries mentioned above.
- The **scripts to train models** are all located in the **TrainingScripts** folder in the **Root Directory**. Similarly, the **scripts to test models** are located in the **TestingScripts** folder. Additionally, all the **NEAT Config Files** are located in the **ConfigFiles** folder.
- To test a pre-existing model, open the testing script of the game you want to test and run the code by clicking on the **Code Runner** button or open the built-in terminal in VSCode/IDE and run the command shown below. The example below is to run and test the model for the Bipedal Walker game.
```Python
python TestingScripts/TestBidepalWalker.py
```
- To train your own model, firstly, delete the pickle file of the corresponding game that you want to train from the **Winners** folder. Open the training script of the game you want to train and run the code by clicking on the **Code Runner** button or open the built-in terminal in VSCode/IDE and run the command shown below. The example below is to train the model for the Bipedal Walker game.
```Python
python TrainingScripts/BidepalWalker.py
```
- Once the training process is done, follow **Step 5** to test the model.
- Enjoy using this project!
