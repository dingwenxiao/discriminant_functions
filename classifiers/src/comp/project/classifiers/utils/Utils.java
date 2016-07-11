package comp.project.classifiers.utils;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import Jama.Matrix;

public class Utils {

	public static double[][] getPointsFromFiles(String file, int length, int dim) {
		String line = null;
		double[][] points = new double[length][dim];
		try {
			FileReader fileReader = new FileReader(file);
			BufferedReader bufferedReader = new BufferedReader(fileReader);
			int count = 0;
			while ((line = bufferedReader.readLine()) != null) {
				if(line.contains("x"))
					continue;
				
				double[] point = getPointFromStr(line);
				for(int i=0; i<dim; i++) {
					points[count][i] = point[i];
				}
				count++;
			}
			bufferedReader.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (SecurityException e) {
			e.printStackTrace();
		} catch (IOException ex) {
			System.out.print(ex);
		}
		return points;
	}

	private static double[] getPointFromStr(String line) {
		String[] numString = line.split(",");
		double[] res = new double[numString.length];
		for (int i = 0; i < numString.length; i++) {
			res[i] = Double.valueOf(numString[i]);
		}
		return res;
	}
	
	public static Matrix transToMatrix(double[] x_array) {
		double[][] xx = {x_array};//{{1,2,3}}
		Matrix x = new Matrix(xx).transpose();
		return x;
	}
}
