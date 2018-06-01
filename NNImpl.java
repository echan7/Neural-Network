/**
 * The main class that handles the entire network
 * Has multiple attributes each with its own use
 * 
 */

import java.util.*;


public class NNImpl{
	public ArrayList<Node> inputNodes=null;//list of the input layer nodes.
	public ArrayList<Node> hiddenNodes=null;//list of the hidden layer nodes
	public ArrayList<Node> outputNodes=null;// list of the output layer nodes
	
	public ArrayList<Instance> trainingSet=null;//the training set
	
	Double learningRate=1.0; // variable to store the learning rate
	int maxEpoch=1; // variable to store the maximum number of epochs
	
	/**
 	* This constructor creates the nodes necessary for the neural network
 	* Also connects the nodes of different layers
 	* After calling the constructor the last node of both inputNodes and  
 	* hiddenNodes will be bias nodes. 
 	*/
	
	public NNImpl(ArrayList<Instance> trainingSet, int hiddenNodeCount, Double learningRate, int maxEpoch, Double [][]hiddenWeights, Double[][] outputWeights)
	{
		this.trainingSet=trainingSet;
		this.learningRate=learningRate;
		this.maxEpoch=maxEpoch;
		
		//input layer nodes
		inputNodes=new ArrayList<Node>();
		int inputNodeCount=trainingSet.get(0).attributes.size();
		int outputNodeCount=trainingSet.get(0).classValues.size();
		for(int i=0;i<inputNodeCount;i++)
		{
			Node node=new Node(0);
			inputNodes.add(node);
		}
		
		//bias node from input layer to hidden
		Node biasToHidden=new Node(1);
		inputNodes.add(biasToHidden);
		
		//hidden layer nodes
		hiddenNodes=new ArrayList<Node> ();
		for(int i=0;i<hiddenNodeCount;i++)
		{
			Node node=new Node(2);
			//Connecting hidden layer nodes with input layer nodes
			for(int j=0;j<inputNodes.size();j++)
			{
				NodeWeightPair nwp=new NodeWeightPair(inputNodes.get(j),hiddenWeights[i][j]);
				node.parents.add(nwp);
			}
			hiddenNodes.add(node);
		}
		
		//bias node from hidden layer to output
		Node biasToOutput=new Node(3);
		hiddenNodes.add(biasToOutput);
			
		//Output node layer
		outputNodes=new ArrayList<Node> ();
		for(int i=0;i<outputNodeCount;i++)
		{
			Node node=new Node(4);
			//Connecting output layer nodes with hidden layer nodes
			for(int j=0;j<hiddenNodes.size();j++)
			{
				NodeWeightPair nwp=new NodeWeightPair(hiddenNodes.get(j), outputWeights[i][j]);
				node.parents.add(nwp);
			}	
			outputNodes.add(node);
		}	
	}
	
	/**
	 * Get the output from the neural network for a single instance
	 * Return the idx with highest output values. For example if the outputs
	 * of the outputNodes are [0.1, 0.5, 0.2, 0.1, 0.1], it should return 1. If outputs
	 * of the outputNodes are [0.1, 0.5, 0.1, 0.5, 0.2], it should return 3. 
	 * The parameter is a single instance. 
	 */
	
	public int calculateOutputForInstance(Instance inst)
	{
		for(int k =0; k < inst.attributes.size() ; k++){
			inputNodes.get(k).setInput(inst.attributes.get(k));
		}
		for(int i =0; i < hiddenNodes.size();i++){
			hiddenNodes.get(i).calculateOutput();
		}
		for(int j =0; j < outputNodes.size(); j++){
			outputNodes.get(j).calculateOutput();
		}
		
		
		double max = Double.MIN_VALUE;
		int maxIndex = 0;
		max = outputNodes.get(maxIndex).getOutput();
		for(int k = 0; k <outputNodes.size(); k ++){
			if(outputNodes.get(k).getOutput() >= max){
				max = outputNodes.get(k).getOutput();
				maxIndex = k;
			}
		}
		
		return maxIndex;
		// TODO: add code here
	}
	

	
	
	
	/**
	 * Train the neural networks with the given parameters
	 * 
	 * The parameters are stored as attributes of this class
	 */
	
	public void train()
	{
		// TODO: add code here
		for(int i =0; i < maxEpoch; i++){
			for(int j = 0; j < trainingSet.size(); j++){
				ArrayList<Double> hiddenNodesDeltaList = new ArrayList<Double>(); 
				ArrayList<Double> outputDeltaList = new ArrayList<Double>();
				//ArrayList<ArrayList<Double>> weightDeltaList = new ArrayList<ArrayList<Double>>();
				calculateOutputForInstance(trainingSet.get(j));
				
				for(int k = 0; k < trainingSet.get(j).classValues.size();k++){
					double deltaTrain = trainingSet.get(j).classValues.get(k) - outputNodes.get(k).getOutput();
					double deltaOutput = outputNodes.get(k).getOutput()* (1.0 - outputNodes.get(k).getOutput())*deltaTrain;
					outputDeltaList.add(deltaOutput);
				}
				
				
				double deltaSum = 0.0;
				for(int l =0; l < hiddenNodes.size() -1; l++){
					for(int s = 0; s < outputNodes.size(); s++){
						deltaSum += outputNodes.get(s).parents.get(l).weight * outputDeltaList.get(s);
					}
					
					double deltaHidden = hiddenNodes.get(l).getOutput() * (1.0 - hiddenNodes.get(l).getOutput()) * deltaSum;
					hiddenNodesDeltaList.add(deltaHidden);
					
				}
				
				for(int f =0; f < hiddenNodes.size() -1; f++){
					for(int c =0; c<hiddenNodes.get(f).parents.size(); c++){
						hiddenNodes.get(f).parents.get(c).weight = hiddenNodes.get(f).parents.get(c).weight + (learningRate * hiddenNodes.get(f).parents.get(c).node.getOutput() * hiddenNodesDeltaList.get(f));
					}
				}
				
				
				for(int m = 0; m < outputNodes.size(); m++){
					for(int c =0; c < outputNodes.get(m).parents.size(); c++){
						outputNodes.get(m).parents.get(c).weight = outputNodes.get(m).parents.get(c).weight + (learningRate * outputNodes.get(m).parents.get(c).node.getOutput() * outputDeltaList.get(m));
					}
				}
				
			}
		}
	}
}
