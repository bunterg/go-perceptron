package main

import "errors"

// Perceptron network structure
type Perceptron struct {
	input   uint
	weights []float64
	bias    []float64
}

// NewPerceptron creates a perceptron network structure
func NewPerceptron(i uint) *Perceptron {
	weights := make([]float64, i)
	bias := make([]float64, i)
	return &Perceptron{
		input:   i,
		weights: weights,
		bias:    bias,
	}
}

// Train network with given input and result
func (p *Perceptron) Train(in [][]float64, o []float64) {

}

// Exec performs compute
func (p *Perceptron) Exec(in []float64) (float64, error) {
	if uint(len(in)) != p.input {
		return 0, errors.New("Wrong input dimension size")
	}
	return 0, nil
}

// GetWeights returns current Bias
func (p *Perceptron) GetWeights() []float64 {
	return p.weights
}

// GetBias returns current bias
func (p *Perceptron) GetBias() []float64 {
	return p.bias
}

// GetError returns current error
func (p *Perceptron) GetError() float64 {
	return 0
}
