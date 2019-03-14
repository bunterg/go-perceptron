package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"strconv"
	"time"
)

const (
	learningRate    = 0.1
	inputDimension  = 2
	hiddenDimension = 16
	outputDimension = 1
	binaryDimension = 8
)

func main() {
	fileTestInput, err := os.Open("sample/iris_in.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer fileTestInput.Close()
	fileTestOutput, err := os.Open("sample/iris_out.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer fileTestOutput.Close()
	iTest := readerToDataInput(csv.NewReader(fileTestInput))
	oTest := readerToDataOutput(csv.NewReader(fileTestOutput))

	p := NewPerceptron(4, 0.05, func(res float64) float64 {
		out := math.Min(res, 2)
		out = math.Max(res, 0)
		if out > 0 && out < 2 {
			out = 1
		}
		return out
	})
	start := time.Now()
	p.Train(iTest, oTest, 50)
	p.DisplayData()
	t := time.Now()
	fmt.Println(t.Sub(start))
}

func readerToDataInput(r *csv.Reader) [][]float64 {
	res := make([][]float64, 150)
	i := 0
	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatal(err)
		}
		res[i] = make([]float64, len(record))
		for j, r := range record {
			res[i][j], _ = strconv.ParseFloat(r, 64)
		}
		i++
	}
	return res
}

func readerToDataOutput(r *csv.Reader) []float64 {
	res := make([]float64, 150)
	i := 0
	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatal(err)
		}
		res[i], _ = strconv.ParseFloat(record[0], 64)
		i++
	}
	return res
}
