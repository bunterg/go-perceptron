package main

import (
	"log"
	"math"
)

const (
	learningRate    = 0.1
	inputDimension  = 2
	hiddenDimension = 16
	outputDimension = 1
	binaryDimension = 8
)

func main() {
	log.Println("hola mundo rnn")
	largestNumber := uint64(math.Pow(2, binaryDimension))
	idk := 01111111
	log.Println(byte(largestNumber))
	log.Println(idk)
}

// ACTIVATION FUNCTIONS
func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func reLU(x float64) float64 {
	return math.Max(0, x)
}

func tanh(x float64) float64 {
	return math.Tanh(x)
}

//
func sigmoidDerivative(x float64) float64 {
	return x * (1 - x)
}
