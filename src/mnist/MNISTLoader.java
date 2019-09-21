package mnist;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;;

public class MNISTLoader {
	private List<List<Double>> data = new ArrayList<>();
	/**
	 * Loads the numbers from the MNIST file and enters them into a list which is then appended to the data list.
	 * 
	 * Each image is represented by an array of numbers between 0 and 255, and each array is added to the data list.
	 * 
	 * @param  fileName		  - The path of the MNIST file.
	 * @param  numberOfImages - The number of images to load from the file.
	 * @return data			  - A 2d list where each entry list is a 1d representation of an image.
	 */
	public List<List<Double>> loadCSV(String fileName, int numberOfImages) {
		if (data.size() > 0) data = new ArrayList<>();			
		try {
			BufferedReader br = new BufferedReader(new FileReader(fileName));
			String line;
			int i = 0;
			while((line = br.readLine()) != null && i < numberOfImages) {
				List<String> stringValues = Arrays.asList(line.split(","));
				List<Double> doubleValues = new ArrayList<>();
				stringValues.stream().mapToDouble(num -> Double.parseDouble(num)).forEach(doubleValues::add);
				data.add(doubleValues);
				i++;
			}
			br.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return data;
	}
	
	/**
	 * The data list is copied and turned into a list of DoubleMatrix2D, where the instances of DoubleMatrix2D represents the image 
	 * as a matrix with a single column and 784 rows. 
	 * 
	 * @return inputs - A list of DoubleMatrix2D.
	 */
	public List<DoubleMatrix2D> getRawInputs(){
		List<DoubleMatrix2D> inputs = new ArrayList<>();
		DoubleMatrix2D matrix;
		for(List<Double> row: data) {
			matrix = new DenseDoubleMatrix2D(MNIST.nPixels, 1);
			for(int i = 1; i < row.size(); i++) {
				matrix.set(i-1, 0, row.get(i));
			}
			inputs.add(matrix);
		}
		return inputs;	
	}
	
	private List<DoubleMatrix2D> normalizedInputs;
	/**
	 * Normalizes the numbers in the data list to be between 0 and 1, and adds them to a list of DoubleMatrix2D called normalizedInputs.
	 * The DoubleMatrix2D are 784 by 1.
	 */
	public void initNormalizedInputs(){
		normalizedInputs = new ArrayList<>();
		DoubleMatrix2D matrix;
		for(List<Double> row: data) {
			matrix = new DenseDoubleMatrix2D(MNIST.nPixels, 1);
			for(int i = 1; i < row.size(); i++) {
				matrix.set(i-1, 0, row.get(i) > 75 ? 1 : 0);
			}
			normalizedInputs.add(matrix.copy());
		}
	}
	
	List<DoubleMatrix2D> targets;
	/**
	 * Initializes the targets list, containing 10x1 DoubleMatrix2D representing the target output of the neural network.
	 * example:
	 * If the target output is 3 the matrix would look like:
	 * m = 	[ 0,
	 * 		  0,
	 * 		  0,
	 * 		  1,
	 * 		  0,
	 * 		  0,
	 * 		  0,
	 * 		  0,
	 * 		  0,
	 * 		  0 ]
	 */
	public void initTargets(){
		targets = new ArrayList<>();
		DoubleMatrix2D matrix;
		for(List<Double> row : data) {
			matrix = new DenseDoubleMatrix2D(MNIST.nLabels, 1);
			matrix.set(row.get(0).intValue(), 0, 1);
			targets.add(matrix);
		}
	}
	
	/**
	 * Returns the list list of target matixes.
	 * @return targets
	 */
	public List<DoubleMatrix2D> getTargets(){
		return targets;
	}

	/**
	 * Returns the list of raw data, with the form List<List<Double>>.
	 * @return data
	 */
	public List<List<Double>> getData() {
		return data;		
	}
	
	/**
	 * Returns the normalized inputs where the images are represented as DoubleMatrix2D instead of List<Double>.
	 * @return normalizedInputs.
	 */
	public List<DoubleMatrix2D> getNormalizedInputs() {
		return normalizedInputs;
	}
	
}

