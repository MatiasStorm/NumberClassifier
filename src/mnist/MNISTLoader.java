package mnist;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;

public class MNISTLoader {
	
	private List<List<Double>> data = new ArrayList<>();
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
		} catch (IOException e) {
			e.printStackTrace();
		}
		return data;
	}
	
	public List<DoubleMatrix2D> getRawInputs(){
		List<DoubleMatrix2D> inputs = new ArrayList<>();
		DoubleMatrix2D matrix;
		for(List<Double> row: data) {
			matrix = new DenseDoubleMatrix2D(28 * 28, 1);
			for(int i = 1; i < row.size(); i++) {
				matrix.set(i-1, 0, row.get(i));
			}
			inputs.add(matrix);
		}
		return inputs;	
	}
	
	private List<DoubleMatrix2D> normalizedInputs;
	public void initNormalizedInputs(){
		normalizedInputs = new ArrayList<>();
		DoubleMatrix2D matrix;
		for(List<Double> row: data) {
			matrix = new DenseDoubleMatrix2D(28 * 28, 1);
			for(int i = 1; i < row.size(); i++) {
//				matrix.set(i-1, 0, row.get(i) / 255.0);
				matrix.set(i-1, 0, row.get(i) > 75 ? 1 : 0);
			}
			normalizedInputs.add(matrix.copy());
		}
	}
	
	List<DoubleMatrix2D> targets;
	public void initTargets(){
		targets = new ArrayList<>();
		DoubleMatrix2D matrix;
		for(List<Double> row : data) {
			matrix = new DenseDoubleMatrix2D(10, 1);
			matrix.set(row.get(0).intValue(), 0, 1);
			targets.add(matrix);
		}
	}
	
	public List<DoubleMatrix2D> getTargets(){
		return targets;
	}

	public List<List<Double>> getData() {
		return data;		
	}
	
	public List<DoubleMatrix2D> getNormalizedInputs() {
		return normalizedInputs;
	}
	
}








