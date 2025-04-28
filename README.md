# Early Exit In Scene Graph Generation

1. This project has been forked from the original [RelTr Paper GitHub Repository](https://github.com/yrcong/RelTR)
2. To Setup Original RelTr Code refer to this [README.md](https://github.com/yrcong/RelTR/blob/main/README.md).
3. Create a "model weights" folder and download the trained model weight from [Drive Link](https://drive.google.com/drive/folders/16AjFdYgA57YxjjuQt3iWF892tQcPkPoh?usp=sharing).
4. To download the images dataset and the annotation/label files refer to [Drive Link](https://drive.google.com/drive/folders/1rSjJYvZb84bNtn77Nuwdq1oRRSl3_Eyv?usp=sharing). Add the data to a folder "data/vg/".
5. We have added our code files which are: \
	a. **Baseline.ipynb** - Contains baseline RelTr evaluation code. \
	b. **Early-Exit-SGG.ipynb** - Training code with imitation layers for early exit. \
	c. **EarlyExitMetrics.ipynb** - Experiments to evaluate effectiveness of early exit: per-layer and across entropy thresholds. \
	d. **EntropyandExitExperiments.ipynb** - Experiments to evaluate early exit across entropy thresholds, inference times, and evaluate effectiveness of imitation layer. \
	e. **Training Curves.ipynb** - Scripts to clean the RelTr metrics to a more readable format, and generate graphs. \
	f. **Per Layer Metrics.ipynb** - Calculates the RelTr metrics for exit at each decoder layer, using trained imitation layers.