package comp.project.classifiers.imp;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.math.BigDecimal;

import Jama.Matrix;

import comp.project.classifiers.Classifier;
import comp.project.classifiers.utils.Utils;

public class HoKashyap extends Classifier {

	public static double ng = .9;
	static int kMax = 150000;//maximum iterations
	Matrix a;
	Matrix b; 

	@Override
	public void train(double[][] class1, double[][] class2) {
		int dim = class1[0].length;
		Matrix class1M = new Matrix(class1);
		Matrix class1Ym = new Matrix(class1.length, dim + 1, 1);
		class1Ym.setMatrix(0, class1.length - 1, 1, dim, class1M);

		Matrix class2M = new Matrix(class2).times(-1);
		Matrix class2Ym = new Matrix(class1.length, dim + 1, -1);
		class2Ym.setMatrix(0, class2.length - 1, 1,
				dim, class2M);

		Matrix Y = new Matrix(class1.length + class2.length, dim + 1);
		Y.setMatrix(0, class1.length - 1, 0, dim, class1Ym);
		Y.setMatrix(class1.length, class1.length + class2.length - 1, 0, dim,
				class2Ym);
		a = new Matrix(dim + 1, 1, 1);
		Matrix bnext = new Matrix(class1.length + class2.length, 1, 1);
		b = bnext;
		Matrix e = Y.times(a).minus(b);
		while (!greaterThanZero(e)) {
			b = bnext;// update b
			e = Y.times(a).minus(b);
			bnext = b.plus(e.plus(abs(e.copy())).times(ng));
			Matrix first = Y.transpose().times(Y);
			a = first.inverse().times(Y.transpose()).times(bnext);
		}
		b = bnext;
	}

	@Override
	public boolean classify(double[] x_array) {
		Matrix y = transToY(x_array);
		return greaterThanZero(y.transpose().times(a));
	}
	
	private Matrix transToY(double[] x_array) {
		Matrix x = Utils.transToMatrix(x_array);
		Matrix y = new Matrix(x.getRowDimension()+1,x.getColumnDimension(),1);
		y.setMatrix(1,y.getRowDimension()-1,0,y.getColumnDimension()-1,x);
		return y;
	}

	public Matrix abs(Matrix x) {
		for (int i = 0; i < x.getColumnDimension(); i++) {
			for (int j = 0; j < x.getRowDimension(); j++) {
				x.set(j, i, Math.abs(x.get(j, i)));
			}
		}
		return x;
	}

	public boolean greaterThanZero(Matrix x) {
		for (int i = 0; i < x.getColumnDimension(); i++) {
			for (int j = 0; j < x.getRowDimension(); j++) {
				double xx = x.get(j, i);
				BigDecimal bd = new BigDecimal(xx);
				bd.setScale(4, BigDecimal.ROUND_HALF_UP);
				xx = bd.doubleValue();
				if (Math.abs(xx)>0.00001 && xx < 0) {
					return false;
				}
			}
		}
		return true;
	}
	
	public boolean equalToZero(Matrix x) {
		for (int i = 0; i < x.getColumnDimension(); i++) {
			for (int j = 0; j < x.getRowDimension(); j++) {
				double xx = x.get(j, i);
				BigDecimal bd = new BigDecimal(xx);
				bd.setScale(4, BigDecimal.ROUND_HALF_UP);
				xx = bd.doubleValue();
				if (Double.valueOf(Math.abs(xx)).compareTo(Double.valueOf(0.00001))>0) {
					return false;
				}
			}
		}
		return true;
	}

	@Override
	public void drawDiscrFunc(String[] fileNames) {
		try {
			for (int i = 1; i <= fileNames.length; i++) {
				File file = new File(fileNames[i - 1]);

				if (!file.exists()) {
					file.createNewFile();
				}

				FileWriter fileWritter = new FileWriter(file.getName());
				BufferedWriter bufferWritter = new BufferedWriter(fileWritter);
				bufferWritter.write("x2,x" + (i+3) + "\n");
				
				BigDecimal   aiDec   =   new   BigDecimal(a.get(i+3, 0));  
				double   ai   =   aiDec.setScale(4,   BigDecimal.ROUND_HALF_UP).doubleValue();  
				
				if(ai==0) {
					continue;
				}

				for (double x2 = 0; x2 <= 10; x2 = x2 + 0.001) {
					double xi = -0.5*x2+10;
					bufferWritter.write(x2+","+xi+"\n");
				}
				bufferWritter.close();
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public static void main(String[] args) {
		double[][] class1 = Utils.getPointsFromFiles("iris_class1_d.csv", 50, 4);
		double[][] class2 = Utils.getPointsFromFiles("iris_class2_d.csv", 50, 4);

		Classifier hoKashyap = new HoKashyap();
		hoKashyap.train(class1, class2);
		String[] fileNames = {"hk_24d"};
		hoKashyap.drawDiscrFunc(fileNames);
	}
}
