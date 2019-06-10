package neuralnetwork;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import cern.colt.function.DoubleFunction;
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;
import cern.jet.math.Functions;
import cern.jet.random.engine.DRand;

public class NeuralNetwork {
	// Matrix functions:
	private static DoubleFunction activate = new DoubleFunction() {
		public double apply(double x) {return 1 / (1 + Math.exp(-x));}
	};
	private static DoubleFunction dActivate = new DoubleFunction() {
		public double apply(double x) {return x * (1 - x);}
	};
	private static DRand randomEngine = new DRand(1);
	private static DoubleFunction random = new DoubleFunction() {
		public double apply(double x) {return randomEngine.nextDouble() - 0.5;}
	};
	
	
	private static final DoubleFactory2D factory = DoubleFactory2D.dense;
	public DoubleMatrix2D outputWeights, outputBias;
	private int nInputNodes, nOutputNodes, nHiddenLayers;
	public List<DoubleMatrix2D> hiddenWeights = new ArrayList<>();
	public List<DoubleMatrix2D> hiddenBiases = new ArrayList<>();
	List<Integer> hiddenLayers;
	private static final Algebra algebra = new Algebra();
	private double alfa;
	public NeuralNetwork(int _nInputNodes, Integer[] _hiddenLayers, int _nOutputNodes, double _alfa) {
		alfa = _alfa;
		nInputNodes = _nInputNodes;
		nOutputNodes = _nOutputNodes;
		
		hiddenLayers = Arrays.asList(_hiddenLayers);
		nHiddenLayers = hiddenLayers.size();
		
		DoubleMatrix2D weights;
		DoubleMatrix2D bias;
		for(int i = 0; i < nHiddenLayers; i++) {
			int nodes = hiddenLayers.get(i);
			if(i == 0) {
				weights = factory.make(nodes, nInputNodes);
				bias = factory.make(nodes, 1);
			} else{
				weights = factory.make(nodes, hiddenLayers.get(i - 1));
				bias = factory.make(nodes, 1);
			}
			weights.assign(random);
			bias.assign(random);
			hiddenWeights.add(weights);
			hiddenBiases.add(bias);
		}
		outputWeights = factory.make(nOutputNodes, hiddenLayers.get(nHiddenLayers - 1));
		outputWeights.assign(random);
		outputBias = factory.make(nOutputNodes, 1);
		outputBias.assign(random);
	}
	
	
	public NeuralNetwork(String fileName, double _alfa) {
		try {
			alfa = _alfa;
			BufferedReader reader = new BufferedReader(new FileReader(fileName));
			nInputNodes = Integer.parseInt(reader.readLine());
			
			hiddenLayers = new ArrayList<>();
			for(String number : reader.readLine().split(" ")) {
				hiddenLayers.add(Integer.parseInt(number));
			}
			nHiddenLayers = hiddenLayers.size();
			nOutputNodes = Integer.parseInt(reader.readLine());
			
			initializeMatrixes();
			
			for(int j = 0; j < nHiddenLayers; j++) {
				readMatrixFromFile(hiddenWeights.get(j), reader);
				readMatrixFromFile(hiddenBiases.get(j), reader);
			}
			readMatrixFromFile(outputWeights, reader);
			readMatrixFromFile(outputBias, reader);
			reader.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	
	private void initializeMatrixes() {		
		DoubleMatrix2D weights, bias;
		for(int i = 0; i < nHiddenLayers; i++) {
			int nodes = hiddenLayers.get(i);
			if(i == 0) {
				weights = factory.make(nodes, nInputNodes);
				bias = factory.make(nodes, 1);
			} else{
				weights = factory.make(nodes, hiddenLayers.get(i - 1));
				bias = factory.make(nodes, 1);
			}
			hiddenWeights.add(weights);
			hiddenBiases.add(bias);
		}
		
		outputWeights = factory.make(nOutputNodes, hiddenLayers.get(nHiddenLayers - 1));
		outputBias = factory.make(nOutputNodes, 1);
	}
	
	
	private void readMatrixFromFile(DoubleMatrix2D matrix, BufferedReader reader) {
		String line;
		try {
			if((line = reader.readLine()) != null) {
				ArrayList<String> stringData = new ArrayList<>(Arrays.asList(line.split(" ")));
				int rows = Integer.parseInt(stringData.remove(0));
				int columns = Integer.parseInt(stringData.remove(0));
				for(int i = 0; i < stringData.size(); i++) {
					int r = Math.floorDiv(i, columns);
					int c = i % columns;
					matrix.set(r, c, Double.parseDouble(stringData.get(i)));
				}
				
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	
	private void writeMatrixToFile(DoubleMatrix2D matrix, PrintWriter writer) {
		writer.print(matrix.rows() + " " + matrix.columns() + " ");
		for(int r = 0; r < matrix.rows(); r++) {
			for (int c = 0; c < matrix.columns(); c++) {
				writer.print(matrix.get(r, c) + " ");
			}
		}
		writer.println();
	}
	
	
	public void saveData(String fileName) {
		try {
			PrintWriter writer = new PrintWriter(fileName);
			writer.println(nInputNodes);
			for(int i = 0; i < nHiddenLayers; i++) {
				writer.print(hiddenLayers.get(i) + " ");				
			}
			writer.println();
			writer.println(nOutputNodes);
			
			// Writing matrixes to file:
			for(int i = 0; i < nHiddenLayers; i++) {
				writeMatrixToFile(hiddenWeights.get(i), writer);
				writeMatrixToFile(hiddenBiases.get(i), writer);
			}
			writeMatrixToFile(outputWeights, writer);
			writeMatrixToFile(outputBias, writer);
			
			writer.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
	
	
	public DoubleMatrix2D feedForward(DoubleMatrix2D input) {
		DoubleMatrix2D guess = input.copy();
		
		for(int i = 0; i < nHiddenLayers; i++) {
			guess = algebra.mult(hiddenWeights.get(i), guess);
			guess.assign(hiddenBiases.get(i), Functions.plus);
			guess.assign(activate);
		}
		guess = algebra.mult(outputWeights, guess);
		guess.assign(outputBias, Functions.plus);
		guess.assign(activate);	
		return guess;
	}
	
	
	public void train(DoubleMatrix2D input, DoubleMatrix2D target) {
		// Feed forward:
		DoubleMatrix2D hiddenGuess = input.copy();
		List<DoubleMatrix2D> hiddenGuesses = new ArrayList<>();
		
		for(int i = 0; i < nHiddenLayers; i++) {
			hiddenGuess = algebra.mult(hiddenWeights.get(i), hiddenGuess);
			hiddenGuess.assign(hiddenBiases.get(i), Functions.plus);
			hiddenGuess.assign(activate);
			hiddenGuesses.add(hiddenGuess);
		}
		DoubleMatrix2D outputGuess = algebra.mult(outputWeights, hiddenGuesses.get(hiddenGuesses.size() - 1));
		outputGuess.assign(outputBias, Functions.plus);
		outputGuess.assign(activate);
		
		
		// Back propagation:
		DoubleMatrix2D outputError = target.copy();
		outputError.assign(outputGuess, Functions.minus);
		
		// Output Layer:
		DoubleMatrix2D dBiasOutput = outputGuess.assign(dActivate).copy();
		dBiasOutput.assign(outputError, Functions.mult);
		dBiasOutput.assign(Functions.mult(alfa));
		outputBias.assign(dBiasOutput, Functions.plus); // Update output bias
		DoubleMatrix2D dWeightsOutput = algebra.mult(dBiasOutput, algebra.transpose(hiddenGuesses.get(hiddenGuesses.size() - 1)).copy());
		outputWeights.assign(dWeightsOutput, Functions.plus); // update output weights
		
		// Hidden layers:
		DoubleMatrix2D hiddenError, hiddenInput, dBias, dWeights, nextWeights;
		for(int i = nHiddenLayers - 1; i >= 0; i--) {
			nextWeights = i == hiddenGuesses.size() - 1 ? outputWeights : hiddenWeights.get(i + 1); // The weights of the layer after the current layer.
			hiddenError = algebra.mult(algebra.transpose(nextWeights).copy(), outputError); // The error of the layer.

			hiddenInput = i == 0 ? input.copy() : hiddenGuesses.get(i-1); // The input to the current layer
			
			dBias = (hiddenGuesses.get(i)).copy();
			dBias = dBias.assign(dActivate); // Differentiating the Output.
			dBias.assign(hiddenError, Functions.mult); // Multiplying the error with the guess made by the layer.
			dBias.assign(Functions.mult(alfa)); // Multiplying with the learning rate.
			(hiddenBiases.get(i)).assign(dBias, Functions.plus); // Update hidden Bias.
			dWeights = algebra.mult(dBias, algebra.transpose(hiddenInput.copy())); // Multiplying dBias with the input to the get gradient of the weights.
			(hiddenWeights.get(i)).assign(dWeights, Functions.plus); // Update hidden weights							
			
			outputError = hiddenError.copy(); // Updating the outputError, so it is ready for the next iteration.
		}	
	}
}
