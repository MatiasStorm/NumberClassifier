package neuralnetwork;
import java.io.BufferedReader;
import java.io.File;
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
import cern.jet.random.Uniform;
import cern.jet.random.engine.DRand;

public class NeuralNetwork {
	private int nInputNodes, nHiddenNodes, nOutputNodes;
	private double alfa;
	private static final DoubleFactory2D factory = DoubleFactory2D.dense;
	private static final Algebra algebra = new Algebra();
	public DoubleMatrix2D weightsIH, weightsHO, biasH, biasO;
	
	// Simple Matrix functions:
	private static DoubleFunction activate = new DoubleFunction() {
		public double apply(double x) {return 1 / (1 + Math.exp(-x));}
	};
	private static DoubleFunction dActivate = new DoubleFunction() {
		public double apply(double x) {return x * (1 - x);}
	};
	private static DRand randomEngine = new DRand(1);
	private static DoubleFunction random = new DoubleFunction() {
//		public double apply(double x) {return Uniform.staticNextDoubleFromTo(-1, 1);}
		public double apply(double x) {return randomEngine.nextDouble() - 0.5;}
	};
	
	private void initializeMatrixes() {
		weightsIH = factory.make(nHiddenNodes, nInputNodes);
		weightsHO = factory.make(nOutputNodes, nHiddenNodes);
		biasH = factory.make(nHiddenNodes, 1);
		biasO = factory.make(nOutputNodes, 1);
	}
	
	private void randomizeMatrixes() {
		weightsIH.assign(random);
		weightsHO.assign(random);
		biasH.assign(random);
		biasO.assign(random);
	}
	
	public NeuralNetwork(int _nInputNodes, int _nHiddenNodes, int _nOutputNodes, double _alfa) {
		alfa = _alfa;
		nInputNodes = _nInputNodes;
		nHiddenNodes = _nHiddenNodes;
		nOutputNodes = _nOutputNodes;
		initializeMatrixes();
		randomizeMatrixes();
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
	
	public NeuralNetwork(String fileName) {
		try {
			BufferedReader reader = new BufferedReader(new FileReader(fileName));
			nInputNodes = Integer.parseInt(reader.readLine());
			nHiddenNodes = Integer.parseInt(reader.readLine());
			nOutputNodes = Integer.parseInt(reader.readLine());
			initializeMatrixes();
			readMatrixFromFile(weightsIH, reader);
			readMatrixFromFile(weightsHO, reader);
			readMatrixFromFile(biasH, reader);
			readMatrixFromFile(biasO, reader);
			reader.close();
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
		System.out.println(weightsIH.toString());
		try {
			PrintWriter writer = new PrintWriter(fileName);
			writer.println(nInputNodes);
			writer.println(nHiddenNodes);
			writer.println(nOutputNodes);
			
			writeMatrixToFile(weightsIH, writer);
			writeMatrixToFile(weightsHO, writer);
			writeMatrixToFile(biasH, writer);
			writeMatrixToFile(biasO, writer);
			
			writer.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public DoubleMatrix2D feedForward(DoubleMatrix2D input) {
		DoubleMatrix2D hidden, output;
		// Hidden layer
		hidden = algebra.mult(weightsIH, input);
		hidden.assign(biasH, Functions.plus);
		hidden.assign(activate);
		
		// Output layer
		output = algebra.mult(weightsHO, hidden);
		output.assign(biasO, Functions.plus);
		output.assign(activate);		
		return output;
	}
	
	public void train(DoubleMatrix2D input, DoubleMatrix2D target) {
		DoubleMatrix2D hidden, output, outputError, hiddenError, dWeightsHO;
		// Feed forward:
		// Hidden layer
		hidden = algebra.mult(weightsIH, input);
		hidden.assign(biasH, Functions.plus);
		hidden.assign(activate);
		// Output layer
		output = algebra.mult(weightsHO, hidden);
		output.assign(biasO, Functions.plus);
		output.assign(activate);
		
		// Back propagation:
		// Errors
		outputError = target.copy();
		outputError.assign(output, Functions.minus);
		hiddenError = algebra.mult(algebra.transpose(weightsHO).copy(), outputError);
		// Hidden -> output
		DoubleMatrix2D dOutput = output.assign(dActivate).copy();
		dWeightsHO = dOutput.assign(outputError, Functions.mult);
		dWeightsHO.assign(Functions.mult(alfa));
		biasO.assign(dWeightsHO, Functions.plus); // Update output bias
		dWeightsHO = algebra.mult(dWeightsHO, algebra.transpose(hidden).copy());
		weightsHO.assign(dWeightsHO, Functions.plus); // update output weights
		
		// Input -> Hidden
		DoubleMatrix2D dHidden = hidden.assign(dActivate).copy();
		DoubleMatrix2D dWeightsIH = dHidden.assign(hiddenError, Functions.mult);
		dWeightsIH.assign(Functions.mult(alfa));
		biasH.assign(dWeightsIH, Functions.plus); // Update hidden Bias
		dWeightsIH = algebra.mult(dWeightsIH, algebra.transpose(input.copy()));
		weightsIH.assign(dWeightsIH, Functions.plus); // Update hidden weights
	}
	
	public void trainBatch(DoubleMatrix2D[] inputs, DoubleMatrix2D[] targets, int batchSize) {
		
		
	}
	
	
	
	
	
	
	
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	public List<DoubleMatrix2D> hiddenWeights = new ArrayList<>();
	public List<DoubleMatrix2D> hiddenBiases = new ArrayList<>();
	public DoubleMatrix2D outputWeights, outputBias;
	public NeuralNetwork(int _nInputNodes, int[] hiddenLayers, int _nOutputNodes, double _alfa) {
		alfa = _alfa;
		nInputNodes = _nInputNodes;
		nOutputNodes = _nOutputNodes;
		
		DoubleMatrix2D weights;
		DoubleMatrix2D bias;
		for(int i = 0; i < hiddenLayers.length; i++) {
			int nodes = hiddenLayers[i];
			if(i == 0) {
				weights = factory.make(nodes, nInputNodes);
				bias = factory.make(nodes, 1);
			} else{
				weights = factory.make(nodes, hiddenLayers[i - 1]);
				bias = factory.make(nodes, 1);
			}
			weights.assign(random);
			bias.assign(random);
			hiddenWeights.add(weights);
			hiddenBiases.add(bias);
		}
		outputWeights = factory.make(nOutputNodes, hiddenLayers[hiddenLayers.length - 1]);
		outputWeights.assign(random);
		outputBias = factory.make(nOutputNodes, 1);
		outputBias.assign(random);
	}
	
	public DoubleMatrix2D feedForward2(DoubleMatrix2D input) {
		DoubleMatrix2D guess = input.copy();
		
		for(int i = 0; i < hiddenWeights.size(); i++) {
			guess = algebra.mult(hiddenWeights.get(i), guess);
			guess.assign(hiddenBiases.get(i), Functions.plus);
			guess.assign(activate);
		}
		
		guess = algebra.mult(outputWeights, guess);
		guess.assign(outputBias, Functions.plus);
		guess.assign(activate);	
		return guess;
	}
	
	public void train2(DoubleMatrix2D input, DoubleMatrix2D target) {
		// Feed forward:
		DoubleMatrix2D hiddenGuess = input.copy();
		List<DoubleMatrix2D> hiddenGuesses = new ArrayList<>();
		
		for(int i = 0; i < hiddenWeights.size(); i++) {
			hiddenGuess = algebra.mult(hiddenWeights.get(i), hiddenGuess);
			hiddenGuess.assign(hiddenBiases.get(i), Functions.plus);
			hiddenGuess.assign(activate);
			hiddenGuesses.add(hiddenGuess);
		}
		DoubleMatrix2D outputGuess = algebra.mult(outputWeights, hiddenGuesses.get(hiddenGuesses.size() - 1));
		outputGuess.assign(outputBias, Functions.plus);
		outputGuess.assign(activate);
		
		
		///// Back propagation: /////
		
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
		DoubleMatrix2D hiddenError, preInput, dBias, dWeights, nextWeights;
		for(int i = hiddenGuesses.size() - 1; i >= 0; i--) {
			nextWeights = i == hiddenGuesses.size() - 1 ? outputWeights : hiddenWeights.get(i + 1); // The weights of the layer after the current layer.
			hiddenError = algebra.mult(algebra.transpose(nextWeights).copy(), outputError); // The error of the layer, output error is the error of the output.
			
			outputError = hiddenError.copy(); // Updating the outputError, so it is ready for the next iteration.
			preInput = i == 0 ? input.copy() : hiddenGuesses.get(i-1); // The input to the current layer
			
			dBias = (hiddenGuesses.get(i)).copy();
			dBias = dBias.assign(dActivate); // Differentiating the Output.
			dBias.assign(hiddenError, Functions.mult); // Multiplying the error with the guess made by the layer
			dBias.assign(Functions.mult(alfa)); // Multiplying with the learning rate
			(hiddenBiases.get(i)).assign(dBias, Functions.plus); // Update hidden Bias
			
			dWeights = algebra.mult(dBias, algebra.transpose(preInput.copy())); // Multiplying dBias with the input to the get gradient of the weights of the layer.
			(hiddenWeights.get(i)).assign(dWeights, Functions.plus); // Update hidden weights							
		}
		
	}
	
	
	
//	// Back propagation:
//	// Errors
//	outputError = target.copy();
//	outputError.assign(output, Functions.minus);
//	hiddenError = algebra.mult(algebra.transpose(weightsHO).copy(), outputError);
//	// Hidden -> output
//	DoubleMatrix2D dOutput = output.assign(dActivate).copy();
//	dWeightsHO = dOutput.assign(outputError, Functions.mult);
//	dWeightsHO.assign(Functions.mult(alfa));
//	biasO.assign(dWeightsHO, Functions.plus); // Update output bias
//	dWeightsHO = algebra.mult(dWeightsHO, algebra.transpose(hidden).copy());
//	weightsHO.assign(dWeightsHO, Functions.plus); // update output weights
//	
//	// Input -> Hidden
//	DoubleMatrix2D dHidden = hidden.assign(dActivate).copy();
//	DoubleMatrix2D dWeightsIH = dHidden.assign(hiddenError, Functions.mult);
//	dWeightsIH.assign(Functions.mult(alfa));
//	biasH.assign(dWeightsIH, Functions.plus); // Update hidden Bias
//	dWeightsIH = algebra.mult(dWeightsIH, algebra.transpose(input.copy()));
//	weightsIH.assign(dWeightsIH, Functions.plus); // Update hidden weights
	
	
	
}

























