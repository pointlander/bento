// Copyright 2022 The Bento Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"os/signal"
	"sort"
	"strings"
	"syscall"
	"time"

	"github.com/pointlander/datum/mnist"
	"github.com/pointlander/gradient/tf32"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

// Model is the machine learning model
func Model(inference bool, set, others tf32.Set) tf32.Meta {
	rnd := rand.New(rand.NewSource(1))
	concat := tf32.B(Concat)
	average := tf32.U(AverageRows)
	awareness := tf32.U(Awareness)
	dropout := tf32.U(func(k tf32.Continuation, node int, a *tf32.V) bool {
		size, width := len(a.X), a.S[0]
		c, drops, factor := tf32.NewV(a.S...), make([]int, width), float32(1)/(1-.1)
		for i := range drops {
			if rnd.Float64() > .1 {
				drops[i] = 1
			}
		}
		c.X = c.X[:cap(c.X)]
		for i := 0; i < size; i += width {
			for j, ax := range a.X[i : i+width] {
				if drops[j] == 1 {
					c.X[i+j] = ax * factor
				}
			}
		}
		if k(&c) {
			return true
		}
		for i := 0; i < size; i += width {
			for j := range a.D[i : i+width] {
				if drops[j] == 1 {
					a.D[i+j] += c.D[i+j]
				}
			}
		}
		return false
	})
	if inference {
		dropout = tf32.U(func(k tf32.Continuation, node int, a *tf32.V) bool {
			return k(a)
		})
	}

	layers := make([]tf32.Meta, 0, 8)
	layers = append(layers, dropout(awareness(tf32.Everett(tf32.Add(tf32.Mul(set.Get("a1"), concat(set.Get("position"), others.Get("input"))), set.Get("b1"))))))
	for i := 1; i < Layers; i++ {
		layers = append(layers, dropout(awareness(tf32.Everett(tf32.Add(tf32.Mul(set.Get(fmt.Sprintf("a%d", i+1)), layers[i-1]), set.Get(fmt.Sprintf("b%d", i+1)))))))
	}
	return average(tf32.Add(tf32.Mul(set.Get(fmt.Sprintf("a%d", Layers+1)), layers[Layers-1]), set.Get(fmt.Sprintf("b%d", Layers+1))))
}

const (
	// BatchSize is the size of a batch
	BatchSize = 128
	// B1 exponential decay of the rate for the first moment estimates
	B1 = 0.9
	// B2 exponential decay rate for the second-moment estimates
	B2 = 0.999
	// WidthMultiplier is the width multiplier for the model
	WidthMultiplier = 2
	// Layers is the number of layers
	Layers = 1
)

const (
	// StateM is the state for the mean
	StateM = iota
	// StateV is the state for the variance
	StateV
	// StateTotal is the total number of states
	StateTotal
)

// Position is a position
type Position struct {
	X         int
	Y         int
	Positions []int
}

// SelectPositions selects the positions of input data
func SelectPositions(width, height int, positions []Position) {
	w, h := width/7, height/7
	s := 0
	for k := 0; k < height; k += h {
		for j := 0; j < width; j += w {
			set, index := positions[s], 0
			for y := 0; y < h; y++ {
				for x := 0; x < w; x++ {
					x := (j + x + width) % width
					y := (k + y + height) % height
					set.Positions[index] = x + y*width
					index++
				}
			}
			s++
		}
	}
}

// Concat concats two tensors
func Concat(k tf32.Continuation, node int, a, b *tf32.V) bool {
	if len(a.S) != 2 || len(b.S) != 2 {
		panic("tensor needs to have two dimensions")
	}
	if a.S[1] != b.S[1] {
		panic("dimensions are not the same")
	}
	c := tf32.NewV(a.S[0]+b.S[0], a.S[1])
	for i := 0; i < a.S[1]; i++ {
		for j := 0; j < a.S[0]; j++ {
			c.X = append(c.X, a.X[i*a.S[0]+j])
		}
		for j := 0; j < b.S[0]; j++ {
			c.X = append(c.X, b.X[i*b.S[0]+j])
		}
	}
	if k(&c) {
		return true
	}
	for i := 0; i < a.S[1]; i++ {
		for j := 0; j < a.S[0]; j++ {
			a.D[i*a.S[0]+j] += c.D[i*c.S[0]+j]
		}
		for j := 0; j < b.S[0]; j++ {
			b.D[i*b.S[0]+j] += c.D[i*c.S[0]+j+a.S[0]]
		}
	}
	return false
}

// AverageRows averages the rows of a tensor
func AverageRows(k tf32.Continuation, node int, a *tf32.V) bool {
	size, width, n := len(a.X), a.S[0], float32(a.S[1])
	c := tf32.NewV(width)
	c.X = c.X[:cap(c.X)]
	for i := 0; i < size; i += width {
		for j, ax := range a.X[i : i+width] {
			c.X[j] += ax
		}
	}
	for i := 0; i < width; i++ {
		c.X[i] /= n
	}
	if k(&c) {
		return true
	}
	for i := 0; i < size; i += width {
		for j := range a.D[i : i+width] {
			a.D[i+j] += c.D[j] / n
		}
	}
	return false
}

// Awareness computes a self awareness function
func Awareness(k tf32.Continuation, node int, a *tf32.V) bool {
	size, width, n := len(a.X), a.S[0], float32(a.S[1])
	averages, squared := make([]float32, width), make([]float32, width)
	for i := 0; i < size; i += width {
		for j, ax := range a.X[i : i+width] {
			averages[j] += ax
			squared[j] += ax * ax
		}
	}
	for i := 0; i < width; i++ {
		averages[i] /= n
		squared[i] /= n
	}
	c := tf32.NewV(a.S[0]+2*width, a.S[1])
	for i := 0; i < a.S[1]; i++ {
		for j := 0; j < a.S[0]; j++ {
			c.X = append(c.X, a.X[i*a.S[0]+j])
		}
		for j := 0; j < width; j++ {
			c.X = append(c.X, averages[j])
		}
		for j := 0; j < width; j++ {
			c.X = append(c.X, squared[j]-averages[j]*averages[j])
		}
	}
	if k(&c) {
		return true
	}
	for i := 0; i < a.S[1]; i++ {
		for j := 0; j < a.S[0]; j++ {
			a.D[i*a.S[0]+j] += c.D[i*c.S[0]+j]
			a.D[i*a.S[0]+j] += c.D[i*c.S[0]+j+a.S[0]] / n
			a.D[i*a.S[0]+j] += c.D[i*c.S[0]+j+2*a.S[0]] * 2 * (a.X[i*a.S[0]+j] - averages[j]) / n
		}
	}
	return false
}

var (
	// FlagSet is a weight set file
	FlagSet = flag.String("set", "", "weight set file")
)

func main() {
	// 8474 10000 without awareness
	// 8924 10000 with average awareness
	// 8989 10000 with average and variance awareness
	// 9102 10000 with dropout
	// 8825 10000 with tags on middle layer
	// 9082 10000 with awareness on the input
	// 9155 10000 with relu on first layer
	// 9325 10000 removed TanH from output layer
	// 9599 10000 three layers
	// 9662 10000 four layers
	// 9679 10000 double width
	flag.Parse()

	images, err := mnist.Load()
	if err != nil {
		panic(err)
	}

	width, size := 16, 49
	hidden := 2 * width
	selections := make([]Position, size)
	for i := range selections {
		selections[i].Positions = make([]int, width)
	}
	SelectPositions(images.Train.Width, images.Train.Height, selections)

	if *FlagSet != "" {
		others := tf32.NewSet()
		others.Add("input", width, size)
		for _, w := range others.Weights {
			w.X = w.X[:cap(w.X)]
		}
		inputs := others.ByName["input"]

		set := tf32.NewSet()
		_, _, err = set.Open(*FlagSet)
		if err != nil {
			panic(err)
		}

		model := Model(true, set, others)

		correct := 0
		type Result struct {
			Probability float32
			Index       int
		}
		for i := range images.Test.Images {
			histogram := make([]Result, 10)
			image := images.Test.Images[i]

			for j := range inputs.X {
				inputs.X[j] = 0
			}
			for j, set := range selections {
				for i, value := range set.Positions {
					inputs.X[j*width+i] = float32(image[value]) / 255
				}
			}

			model(func(a *tf32.V) bool {
				for j := 0; j < 10; j++ {
					histogram[j].Probability = a.X[j]
					histogram[j].Index = j
				}
				return true
			})

			sort.Slice(histogram, func(i, j int) bool {
				return histogram[i].Probability > histogram[j].Probability
			})

			if histogram[0].Index == int(images.Test.Labels[i]) {
				correct++
			}
		}

		fmt.Println(correct, len(images.Test.Images))
		return
	}

	c, halt := make(chan os.Signal), false
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-c
		halt = true
	}()

	rnd := rand.New(rand.NewSource(1))

	others := tf32.NewSet()
	others.Add("input", width, size)
	others.Add("output", 10, 1)
	for _, w := range others.Weights {
		w.X = w.X[:cap(w.X)]
	}

	set := tf32.NewSet()
	set.Add("position", width, size)
	set.Add("a1", hidden, WidthMultiplier*hidden)
	set.Add("b1", WidthMultiplier*hidden, 1)
	for i := 1; i < Layers; i++ {
		set.Add(fmt.Sprintf("a%d", i+1), WidthMultiplier*2*3*hidden, WidthMultiplier*hidden)
		set.Add(fmt.Sprintf("b%d", i+1), WidthMultiplier*hidden, 1)
	}
	set.Add(fmt.Sprintf("a%d", Layers+1), WidthMultiplier*2*3*hidden, 10)
	set.Add(fmt.Sprintf("b%d", Layers+1), 10, 1)

	for _, w := range set.Weights {
		if strings.HasPrefix(w.N, "b") {
			w.X = w.X[:cap(w.X)]
			w.States = make([][]float32, StateTotal)
			for i := range w.States {
				w.States[i] = make([]float32, len(w.X))
			}
			continue
		}
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		for i := 0; i < cap(w.X); i++ {
			w.X = append(w.X, float32(rnd.NormFloat64()*factor))
		}
		w.States = make([][]float32, StateTotal)
		for i := range w.States {
			w.States[i] = make([]float32, len(w.X))
		}
	}

	model := Model(false, set, others)
	cost := tf32.Quadratic(model, others.Get("output"))

	i, total, u, start := 0, float32(0), 0.0, time.Now()
	eta := float32(.001)
	pow := func(x float32) float32 {
		y := math.Pow(float64(x), u)
		if math.IsNaN(y) || math.IsInf(y, 0) {
			return 0
		}
		return float32(y)
	}
	points := make(plotter.XYs, 0, 8)
	for i < 40*len(images.Train.Images) {
		index := rnd.Intn(len(images.Train.Images))
		image := images.Train.Images[index]

		inputs, outputs := others.ByName["input"], others.ByName["output"]
		for j := range inputs.X {
			inputs.X[j] = 0
		}
		for j := range outputs.X {
			outputs.X[j] = 0
		}

		for j, set := range selections {
			for i, value := range set.Positions {
				inputs.X[j*width+i] =
					float32(image[value]) / 255
			}
		}
		outputs.X[int(images.Train.Labels[index])] = 1

		total += tf32.Gradient(cost).X[0]

		if i%BatchSize == 0 {
			u++
			b1, b2 := pow(B1), pow(B2)
			for j, w := range set.Weights {
				for k, d := range w.D {
					g := d / BatchSize
					m := B1*w.States[StateM][k] + (1-B1)*g
					v := B2*w.States[StateV][k] + (1-B2)*g*g
					w.States[StateM][k] = m
					w.States[StateV][k] = v
					mhat := m / (1 - b1)
					vhat := v / (1 - b2)
					set.Weights[j].X[k] -= eta * mhat / (float32(math.Sqrt(float64(vhat))) + 1e-8)
				}
			}
			total /= BatchSize
			end := time.Since(start)
			fmt.Println(i, total, end)
			set.Zero()
			others.Zero()
			start = time.Now()

			if halt || math.IsNaN(float64(total)) {
				fmt.Println(total)
				break
			}
			points = append(points, plotter.XY{X: float64(i), Y: float64(total)})

			if i%(BatchSize*10) == 0 {
				set.Save(fmt.Sprintf("%d_set.w", i), total, i)
			}
			total = 0
		}

		i++
	}

	p := plot.New()

	p.Title.Text = "epochs vs cost"
	p.X.Label.Text = "epochs"
	p.Y.Label.Text = "cost"

	scatter, err := plotter.NewScatter(points)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "cost.png")
	if err != nil {
		panic(err)
	}

	set.Save("set.w", 0, 0)
}
