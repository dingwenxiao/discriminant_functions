package comp.project.testers;

import comp.project.classifiers.Classifier;
import comp.project.classifiers.imp.Fisher;
import comp.project.classifiers.imp.HoKashyap;
import comp.project.classifiers.imp.KNN;
import comp.project.classifiers.imp.Quadratic;
import comp.project.classifiers.utils.Utils;

public class Kfold implements TestInterface {

	static final int K = 5;

	@Override
	public double testing(double[][] class1, double[][] class2,
			Classifier classifer) {
		int totalNum = class1.length;
		int dim = class1[0].length;
		int oneFoldSize = totalNum / K;

		double[][] class1_test = new double[oneFoldSize][dim];
		double[][] class2_test = new double[oneFoldSize][dim];

		double[][] class1_train = new double[totalNum - oneFoldSize][dim];
		double[][] class2_train = new double[totalNum - oneFoldSize][dim];

		double[] accuracies = new double[K];
		for (int i = 0; i < K; i++) {
			double correct_count = 0;
			for (int t = 0,s=0; t < K; t++) {// s tracks the number of training data
				if (t == i) {// train fold is same with test fold pass
					continue;
				}
				for (int j = 0; j < oneFoldSize; j++) {
					for (int d = 0; d < dim; d++) {
						class1_train[s][d] = class1[t * oneFoldSize + j][d];
						class2_train[s][d] = class2[t * oneFoldSize + j][d];
					}
					s++;
				}
			}
			classifer.train(class1_train, class2_train);

			// testing
			for (int j = 0; j < oneFoldSize; j++) {

				for (int d = 0; d < dim; d++) {
					class1_test[j][d] = class1[i * oneFoldSize + j][d];
					class2_test[j][d] = class2[i * oneFoldSize + j][d];
				}

				if (classifer.classify(class1_test[j])) {
					correct_count++;
				}

				if (!classifer.classify(class2_test[j])) {
					correct_count++;
				}
			}
			accuracies[i] = correct_count / ((double) oneFoldSize * 2);
		}
		return average(accuracies);
	}

	double average(double[] accuracies) {
		double sum = 0;
		for (int i = 0; i < accuracies.length; i++) {
			sum += accuracies[i];
		}
		return sum / (double) accuracies.length;
	}

	public static void main(String[] args) {
		Kfold kf = new Kfold();

		double[][] class1 = Utils.getPointsFromFiles("iris_class1_d.csv", 50, 4);
		double[][] class2 = Utils.getPointsFromFiles("iris_class2_d.csv", 50, 4);

		Classifier knn = new KNN();
		System.out.println(kf.testing(class1, class2, knn));
		
		Classifier qd = new Quadratic();
		System.out.println(kf.testing(class1, class2, qd));
		
		Classifier fisher = new Fisher();
		System.out.println(kf.testing(class1, class2, fisher));
		
		Classifier hoKashyap = new HoKashyap();
		System.out.println(kf.testing(class1, class2, hoKashyap));
	}
}
