package comp.project.classifiers.imp;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.math.BigDecimal;

import comp.project.classifiers.Classifier;
import comp.project.classifiers.utils.Utils;
import Jama.Matrix;

public class Fisher extends Classifier{

	Matrix a;// n by 1
	Matrix m1; // class 1's mean
	Matrix m2;// class 2's mean
	
	@Override
	public void train(double[][] class1, double[][] class2) {
		int num = class1.length;
		int dim = class1[0].length;
		
		double[][] mean1 = new double[dim][1];
		double[][] mean2 = new double[dim][1];
		
		m1 = new Matrix(mean1);
		m2 = new Matrix(mean2);

		for(int i=0; i<num ;i++) {
			double[][] temp1 = {class1[i]};
			m1 = m1.plus(new Matrix(temp1).transpose());
			
			double[][] temp2 = {class2[i]};
			m2 = m2.plus(new Matrix(temp2).transpose());
		}

		m1 = m1.times(1/(double)num);
		m2 = m2.times(1/(double)num);
		
		Matrix s1 = new Matrix(dim,dim);
		Matrix s2 = new Matrix(dim,dim);

		for (int i = 0; i < class1.length; i++) {
			double[][] x1_temp = { class1[i] };//{{1,2,3}}
			Matrix x1 = new Matrix(x1_temp).transpose();
			s1 = s1.plus(x1.minus(m1).times(x1.minus(m1).transpose()));

			double[][] x2_temp = { class2[i] };
			Matrix x2 = new Matrix(x2_temp).transpose();
			s2 = s2.plus(x2.minus(m2).times(x2.minus(m2).transpose()));
		}

		Matrix sw = s1.plus(s2);
		a = sw.inverse().times(m1.minus(m2));
		
		Matrix a0 = a.transpose().times(-1/2).times(m1.plus(m2));
		Matrix temp = new Matrix(a.getRowDimension()+1,1);
		temp.set(0, 0, a0.get(0, 0));
		temp.setMatrix(1, temp.getRowDimension()-1,0,0, a);
		a = temp;
	}

	/**
	 * 
	 * @param double[] x_array   
	 * @return true class1, false class2
	 */
	@Override
	public boolean classify(double[] x_array) {
		double[][] x_trans = new double[][]{x_array};//{{1,1,1}}
		Matrix x = new Matrix(x_trans).transpose();//{{1},{1},{1}}
		Matrix left = x.transpose().times(a).minus(m1.transpose().times(a));
		Matrix right = x.transpose().times(a).minus(m2.transpose().times(a));
		return Math.abs(left.get(0, 0)) < Math.abs(right.get(0, 0));
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
				bufferWritter.write("x1,x" + (i+1) + "\n");
				
				BigDecimal   a0Dec   =   new   BigDecimal(a.get(0, 0));  
				double   a0   =   a0Dec.setScale(4,   BigDecimal.ROUND_HALF_UP).doubleValue(); 
				
				BigDecimal   a1Dec   =   new   BigDecimal(a.get(1, 0));  
				double   a1   =   a1Dec.setScale(4,   BigDecimal.ROUND_HALF_UP).doubleValue();
				
				BigDecimal   aiDec   =   new   BigDecimal(a.get(i+1, 0));  
				double   ai   =   aiDec.setScale(4,   BigDecimal.ROUND_HALF_UP).doubleValue();  
				
				if(ai==0) {
					continue;
				}
				
				for (double x1 = -30; x1 <= 10; x1 = x1 + 0.001) {
					double xi = -(a0*1+a1*x1)/ai;
					bufferWritter.write(x1+","+xi+"\n");
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

		Classifier fisher = new Fisher();
		fisher.train(class1, class2);
		String[] fileNames = {"fisher_12d","fisher_13d","fisher_14d"};
		fisher.drawDiscrFunc(fileNames);
	}
}