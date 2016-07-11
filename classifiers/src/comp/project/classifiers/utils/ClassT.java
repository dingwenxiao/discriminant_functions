package comp.project.classifiers.utils;

import Jama.Matrix;

/**
 * The Class is for class to be classified
 * @author Dingwen
 *
 */
public class ClassT {
	private Matrix covariance;
	private Matrix mean;

	public ClassT(Matrix sigma, Matrix m) {
		this.covariance = sigma;
		this.mean = m;
	}
	
	public Matrix getCovariance() {
		return covariance;
	}

	public void setCovariance(Matrix covariance) {
		this.covariance = covariance;
	}

	public Matrix getMean() {
		return mean;
	}

	public void setMean(Matrix mean) {
		this.mean = mean;
	}

}
