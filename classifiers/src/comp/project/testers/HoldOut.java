package comp.project.testers;

import comp.project.classifiers.Classifier;
import comp.project.classifiers.imp.Fisher;
import comp.project.classifiers.imp.HoKashyap;
import comp.project.classifiers.imp.KNN;
import comp.project.classifiers.imp.Quadratic;
import comp.project.classifiers.utils.Utils;

public class HoldOut implements TestInterface{

	private static final int TRAIN_SIZE = 30;// train size for each class
	
	@Override
	public double testing(double[][] class1, double[][] class2,
			Classifier classifer) {
		int halfNum = class1.length;
		int dim = class1[0].length;

		double[][] class1_train = new double[TRAIN_SIZE][dim];
		double[][] class2_train = new double[TRAIN_SIZE][dim];
		
		double[][] class1_test = new double[class1.length-TRAIN_SIZE][dim];
		double[][] class2_test = new double[class2.length-TRAIN_SIZE][dim];
		
		for(int i=0;i<TRAIN_SIZE;i++){
			for(int j=0;j<dim;j++) {
				class1_train[i][j] = class1[i][j];
				class2_train[i][j] = class2[i][j];
			}
		}
		classifer.train(class1_train, class2_train);
		double correct_num = 0;
		
		for(int i=0;i<halfNum-TRAIN_SIZE;i++){
//			for(int j=0;j<dim;j++) {
//				class1_test[i][j] = class1[i+TRAIN_SIZE][j];
//				
//				class2_test[i][j] = class2[i+TRAIN_SIZE][j];
//			}
			
			if(classifer.classify(class1[i+TRAIN_SIZE])) {
				correct_num++;
			}

			if(!classifer.classify(class2[i+TRAIN_SIZE])) {
				correct_num++;
			}
		}
		
		return correct_num/((halfNum-TRAIN_SIZE)*2);
	}

	
	public static void main(String[] args) {
		double[][] class1 = Utils.getPointsFromFiles("iris_class1_d.csv", 50, 4);
		double[][] class2 = Utils.getPointsFromFiles("iris_class2_d.csv", 50, 4);
		TestInterface holdout = new HoldOut();
		Classifier hoKashyap = new HoKashyap();
		System.out.println(holdout.testing(class1, class2, hoKashyap));
		
		Classifier fisher = new Fisher();
		System.out.println(holdout.testing(class1, class2, fisher));
		
		Classifier qd = new Quadratic();
		System.out.println(holdout.testing(class1, class2, qd));
		
		Classifier knn = new KNN();
		System.out.println(holdout.testing(class1, class2, knn));
	}
}
