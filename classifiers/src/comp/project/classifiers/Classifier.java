package comp.project.classifiers;

public abstract class Classifier {

	
	public abstract void train(double[][] class1, double[][] class2);

	public abstract boolean classify(double[] x_array);

	public abstract void drawDiscrFunc(String[] fileNames);
}
