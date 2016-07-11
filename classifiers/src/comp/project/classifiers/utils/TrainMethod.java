package comp.project.classifiers.utils;

import Jama.Matrix;

public class TrainMethod {

	public static ClassT bayesian(double[][] points, double num, Matrix sigma) {
		int dim = points[0].length;
		Matrix sigma0 = Matrix.identity(dim, dim);
		
		Matrix m0 = new Matrix(dim,1);
		Matrix mn = new Matrix(dim,1);
		
		if (num != 0) {
			double[][] mMLArr = new double[dim][1];
			for (int i = 0; i < num; i++) {
				for(int j=0; j<dim;j++) {
					mMLArr[j][0] += points[i][j];
				}
			}
			for(int j=0; j<dim;j++) {
				mMLArr[j][0] /= num;
			}
			Matrix mML = new Matrix(mMLArr);
			Matrix first = sigma.times(1 / num)
					.times(sigma.times(1 / num).plus(sigma0).inverse())
					.times(m0);
			Matrix second = sigma0.times(
					sigma.times(1 / num).plus(sigma0).inverse()).times(mML);
			mn = first.plus(second);
		}
		return new ClassT(sigma, mn);
	}

	public static ClassT maxiLikeHood(double points[][], double num) {
		int numOfPoints = points.length;
		int dim = points[0].length;

		double[][] estM = new double[dim][1];

		Matrix estMean = new Matrix(estM);
		Matrix var = new Matrix(dim, dim);
		if (num != 0) {
			for (int i = 0; i < num; i++) {
				for (int j = 0; j < dim; j++) {
					estM[j][0] += points[i][j];
				}
			}

			estMean = new Matrix(estM);
			estMean = estMean.times(1 / num);
			for (int i = 0; i < num; i++) {
				double[][] x_array = new double[][]{points[i]};
				Matrix xi = new Matrix(x_array).transpose();
				var = var.plus(xi.minus(estMean).times(
						xi.minus(estMean).transpose()));// Sigma(xi-m)(xi-m)T
			}
			var = var.times(1 / num);
		}
		return new ClassT(var, estMean);
	}
}
