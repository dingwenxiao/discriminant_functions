package comp.project.classifiers.imp;

import java.util.Collections;
import java.util.PriorityQueue;

import comp.project.classifiers.Classifier;

public class KNN extends Classifier {

	public final static int K = 5;
	double[][] class1;
	double[][] class2;
	
	int num;

	@Override
	public void train(double[][] class1, double[][] class2) {
		this.class1 = class1;
		this.class2 = class2;
		this.num = class1.length;
	}

	@Override
	public boolean classify(double[] x_array) {

		PriorityQueue<Double> class1_count = new PriorityQueue<>(K,
				Collections.reverseOrder());
		PriorityQueue<Double> class2_count = new PriorityQueue<>(K,
				Collections.reverseOrder());

		double dis = 0;
		for (int i = 0; i < num; i++) {
			dis = calDis(class1[i], x_array);
			if (class1_count.isEmpty() || class1_count.size() < K) {
				class1_count.offer(dis);
			} else if (class1_count.peek() > dis) {
				class1_count.poll();
				class1_count.offer(dis);
			}
		}

		for (int i = 0; i < num; i++) {
			dis = calDis(class2[i], x_array);
			if (class2_count.isEmpty() || class2_count.size() < K) {
				class2_count.offer(dis);
			} else if (class2_count.peek() > dis) {
				class2_count.poll();
				class2_count.offer(dis);
			}
		}

		int count = 0;
		while (!class1_count.isEmpty() || !class2_count.isEmpty()) {
			if (class1_count.poll() < class2_count.poll()) {
				count++;
			} else {
				count--;
			}
		}
		return count > 0;
	}

	public double calDis(double[] x1, double[] x2) {
		double dis = 0;
		for (int i = 0; i < x1.length; i++) {
			dis += (x1[i] - x2[i]) * (x1[i] - x2[i]);
		}
		return Math.sqrt(dis);
	}

	@Override
	public void drawDiscrFunc(String[] fileName) {
		
	}

}
