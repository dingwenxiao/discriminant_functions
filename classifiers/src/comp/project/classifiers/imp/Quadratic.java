package comp.project.classifiers.imp;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import comp.project.classifiers.Classifier;
import comp.project.classifiers.utils.ClassT;
import comp.project.classifiers.utils.TrainMethod;
import comp.project.classifiers.utils.Utils;
import Jama.Matrix;

public class Quadratic extends Classifier {

	Matrix A;
	Matrix BT;
	Matrix C;

	//classes' parameters evaluated with maximum likelihood
	ClassT class1T;
	ClassT class2T;

	//classes' parameters evaluated with bayesian method
	ClassT class1B;
	ClassT class2B;
	private static final double min = 0.0001;

	@Override
	public void train(double[][] class1, double[][] class2) {
		class1T = TrainMethod.maxiLikeHood(class1, class1.length);
		class2T = TrainMethod.maxiLikeHood(class2, class2.length);
		
		class1B = TrainMethod.bayesian(class1, class1.length, class1T.getCovariance());
		class2B = TrainMethod.bayesian(class2, class2.length,class2T.getCovariance());
		init(class1B, class2B);
	}

	private void init(ClassT class1, ClassT class2) {
		Matrix inSigma1 = class1.getCovariance().inverse();
		Matrix inSigma2 = class2.getCovariance().inverse();
		Matrix trMean1 = class1.getMean().transpose();
		Matrix trMean2 = class2.getMean().transpose();
		A = inSigma2.minus(inSigma1);
		BT = trMean1.times(inSigma1).minus(trMean2.times(inSigma2)).times(2);
		double suffix = 2 * (Math.log(class1.getCovariance().det()
				/ class2.getCovariance().det()));
		Matrix suffixC = new Matrix(1, 1, suffix);
		Matrix secondC = trMean1.times(inSigma1).times(trMean1.transpose());
		C = trMean2.times(inSigma2).times(trMean2.transpose()).minus(secondC)
				.minus(suffixC);
	}

	@Override
	public boolean classify(double[] x_array) {
		Matrix x = Utils.transToMatrix(x_array);
		Matrix first = x.transpose().times(A).times(x);
		Matrix second = BT.times(x);
		Matrix third = C;
		return first.plus(second).plus(third).get(0, 0) > 0;
	}

	@Override
	public void drawDiscrFunc(String[] fileNames) {// 1-2,1-3
		try {

			for (int i = 1; i <= fileNames.length; i++) {
				File file = new File(fileNames[i - 1]);

				if (!file.exists()) {
					file.createNewFile();
				}

				FileWriter fileWritter = new FileWriter(file.getName());
				BufferedWriter bufferWritter = new BufferedWriter(fileWritter);
				bufferWritter.write("x1,x" + (i + 1) + "\n");
				for (double x1 = -30; x1 <= 10; x1 = x1 + 0.01) {

					double a1i = Math.abs(A.get(0, i)) < min ? 0 : A.get(0, i);
					double ai1 = Math.abs(A.get(i, 0)) < min ? 0 : A.get(i, 0);
					double bi = Math.abs(BT.get(0, i)) < min ? 0 : BT.get(0, i);
					double c = Math.abs(C.get(0, 0)) < min ? 0 : C.get(0, 0);

					double b1 = Math.abs(BT.get(0, 0)) < min ? 0 : BT.get(0, 0);
					double a11 = Math.abs(A.get(0, 0)) < min ? 0 : A.get(0, 0);

					double a_fun = Math.abs(A.get(i, i)) < min ? 0 : A
							.get(i, i);
					double b_fun = a1i * x1 + ai1 * x1 + bi;
					double c_fun = a11 * x1 * x1 + b1 * x1 + c;

					String content = genDiscriminantFuncPoints(x1, a_fun,
							b_fun, c_fun, fileNames[i - 1], i);
					if (content != null) {
						bufferWritter.write(content);
					}
				}
				bufferWritter.close();
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private String genDiscriminantFuncPoints(double x1, double a, double b,
			double c, String fileName, int i) {

		double discriminant = b * b - 4 * a * c;
		if (Math.abs(a) <= min) {
			if (b != 0) {
				double root = -c / b;
				return x1 + "," + root + "\n";
			}
		} else if (discriminant == 0) {
			double root = -b / (2 * a);
			return x1 + "," + root + "\n";
		} else if (discriminant > 0) {
			double root1 = (-b + Math.sqrt(discriminant)) / (2 * a);
			double root2 = (-b - Math.sqrt(discriminant)) / (2 * a);
			return x1 + "," + root1 + "\n" + x1 + "," + root2 + "\n";
		}
		return null;
	}

	public static void main(String[] args) {
		int num = 50;
		int dim = 4;
		double[][] class1 = Utils.getPointsFromFiles("iris_class1.csv", num,
				dim);
		double[][] class2 = Utils.getPointsFromFiles("iris_class2.csv", num,
				dim);
		Quadratic ad = new Quadratic();
		ad.train(class1, class2);
		String[] fileNames = { "by_qua_12", "by_qua_13","by_qua_14"};
		ad.drawDiscrFunc(fileNames);

	}

}
