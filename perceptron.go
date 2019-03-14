package main

import (
	"errors"
	"log"
	"math/rand"
)

// Perceptron network structure
type Perceptron struct {
	input       uint
	weights     []float64
	learingRate float64
	activation  func(res float64) float64
}

// NewPerceptron creates a perceptron network structure
func NewPerceptron(i uint, lR float64, activation func(res float64) float64) *Perceptron {
	weights := make([]float64, i+1)
	for i := 0; i < len(weights); i++ {
		weights[i] = rand.Float64()
	}
	return &Perceptron{
		input:       i,
		weights:     weights,
		learingRate: lR,
		activation:  activation,
	}
}

// Train network with given input and result
func (p *Perceptron) Train(in [][]float64, o []float64, logs int) error {
	if len(in) != len(o) {
		return errors.New("Wrong input/output dimension size")
	}
	fails := 0
	for ite, i := range rand.Perm(len(in)) {
		res, _ := p.Predict(in[i])
		if res != o[i] {
			fails++
		}
		p.updateWeights(res, o[i])
		if ite%logs == 0 {
			log.Printf("EPOCH: %d \n", ite)
			log.Printf("Out: %f \n", o[i])
			log.Printf("RES: %f \n", res)
			log.Printf("Succes rate: %f %% \n", float32(fails)/float32(ite+1)*100)
		}

	}
	return nil
}

// Predict performs compute
func (p *Perceptron) Predict(in []float64) (float64, error) {
	if len(in) != len(p.weights)-1 {
		return 0, errors.New("Wrong input dimension size")
	}
	var res float64
	res = p.weights[0]
	for i := 1; i < len(in)+1; i++ {
		res += in[i-1] * p.weights[i]
	}
	return p.activation(res), nil
}

// updateWeights update each weight value
func (p *Perceptron) updateWeights(res float64, o float64) {
	for i := 0; i < len(p.weights); i++ {
		p.weights[i] = p.weights[i] + p.learingRate*(o-res)
	}
}

// GetWeights returns current Bias
func (p *Perceptron) GetWeights() []float64 {
	return p.weights[1:]
}

// GetBias returns current bias
func (p *Perceptron) GetBias() float64 {
	return p.weights[0]
}

// DisplayData shows currents values for some params
func (p *Perceptron) DisplayData() {
	log.Printf("BIAS: %f \n", p.GetBias())
	log.Println("WEIGHTS:")
	log.Println(p.GetWeights())
}
