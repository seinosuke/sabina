# Sabina

Sabina is a machine learning library.  
This gem provides tools for Multi-Layer Perceptrons and Auto-Encoders.  

## Installation

Add this line to your application's Gemfile:

```ruby
gem 'sabina'
```

And then execute:

    $ bundle

Or install it yourself as:

    $ gem install sabina

## Usage

### MultilayerPerceptron

Example of normal usage is shown below.  

```ruby
require 'sabina'

DIM = 2
K = 3
LOOP_NUM = 100

training_data = Sabina::MultilayerPerceptron.load_csv('training_data.csv')

options = {
  :layers => [
    Sabina::Layer::MPInputLayer.new(DIM),
    Sabina::Layer::MPHiddenLayer.new(8),
    Sabina::Layer::MPOutputLayer.new(K)
  ],
  :mini_batch_size => 10,
  :learning_rate => 0.01,
  :training_data => training_data,
}

mp = Sabina::MultilayerPerceptron.new(options)

LOOP_NUM.times do |t|
  mp.learn
  error = mp.error(training_data)
  puts " error : #{error}"
end
```

### AutoEncoder

Example of normal usage is shown below. Use `SparseAutoEncoder` class when the number of input units is less than that of hidden units.

```ruby
require 'sabina'

DIM = 2
LOOP_NUM = 10

original_data = Sabina::AutoEncoder.load_csv('training_data.csv')

options = {
  :layers => [
    Sabina::Layer::AEInputLayer.new(DIM),
    Sabina::Layer::AEHiddenLayer.new(8),
    Sabina::Layer::AEOutputLayer.new(DIM)
  ],
  :mini_batch_size => 10,
  :learning_rate => 0.01,
  :training_data => original_data,
}

sae = Sabina::SparseAutoEncoder.new(options)

LOOP_NUM.times do |t|
  sae.learn
  error = sae.error(original_data)
  puts " error : #{error}"
end
```

### About a training data CSV file format

Examples of a CSV file are shown below.  

```
x0,x1,label
0.8616722150185228,0.7958526101017311,0
0.548524744634457,0.8355704092991548,1
0.2430915120750876,0.6252296416575435,1
0.968877668321639,0.7502385938940324,0
...
```

This is a example for two-dimensional vector data. For example, if you want to input D-dimensional vector data, write `x0,x1,...,x(D-1),label` at the first line.
The column of `label` is used for a cluster id. For example, if there are three clusters in training data, a number at the `label` column will be 0, 1 or 2.  
  
When you prepare a CSV file, load the file as shown below.

```ruby
training_data = Sabina::MultilayerPerceptron.load_csv('training_data.csv')
```

When you use a auto-encoder, load a CSV file as shown below.

```ruby
original_data = Sabina::AutoEncoder.load_csv('training_data.csv')
```

### Configuration

You can set default values by using `Sabina.configure` method. These values could be overwritten by providing an argument.  

```ruby
Sabina.configure do |config|
  config.layers = [
    Sabina::Layer::MPInputLayer.new(2),
    Sabina::Layer::MPHiddenLayer.new(8),
    Sabina::Layer::MPOutputLayer.new(3)
  ]
  config.mini_batch_size = 10
  config.learning_rate = 0.01
  config.training_data = Sabina::MultilayerPerceptron.load_csv('training_data.csv')
end

options = {
  :mini_batch_size => 20
}

mp_01 = Sabina::MultilayerPerceptron.new
mp_02 = Sabina::MultilayerPerceptron.new(options)

mp_01.mini_batch_size # => 10
mp_02.mini_batch_size # => 20
```

### Your own layer class
You can create your own layer class. In the following example below, a rectified linear function is set as an activation function. `@f_` is differentiation of `@f`.

```ruby
class MyHiddenLayer < Sabina::Layer::BaseLayer
  def initialize(size)
    super
    # f(x) = max(0, x)
    @f = ->(x){ x > 0.0 ? x : 0.0 }
    @f_ = ->(x){ x > 0.0 ? 1.0 : 0.0 }
  end
end

options = {
  :layers => [
    Sabina::Layer::MPInputLayer.new(DIM),
    MyHiddenLayer.new(16),
    MyHiddenLayer.new(8),
    Sabina::Layer::MPOutputLayer.new(K)
  ],
  :mini_batch_size => 10,
  :learning_rate => 0.01,
  :training_data => training_data,
}
```

## Examples

These examples require gnuplot version 5.0 or later.

### examples/example_mp_01/

Run [examples/example_mp_01/main.rb](https://github.com/seinosuke/sabina/blob/master/examples/example_mp_01/main.rb).  

![mp_learning_process.gif](https://github.com/seinosuke/sabina/blob/master/examples/example_mp_01/mp_learning_process.gif)

### examples/example_mp_02/

Run [examples/example_mp_02/main.rb](https://github.com/seinosuke/sabina/blob/master/examples/example_mp_02/main.rb).  

![mp_learning_process.gif](https://github.com/seinosuke/sabina/blob/master/examples/example_mp_02/mp_learning_process.gif)

### examples/example_ae_01/

Run [examples/example_ae_01/main.rb](https://github.com/seinosuke/sabina/blob/master/examples/example_ae_01/main.rb).  

![ae_learning_process.gif](https://github.com/seinosuke/sabina/blob/master/examples/example_ae_01/mp_learning_process.gif)


## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/seinosuke/sabina. This project is intended to be a safe, welcoming space for collaboration, and contributors are expected to adhere to the [Contributor Covenant](http://contributor-covenant.org) code of conduct.


## License

The gem is available as open source under the terms of the [MIT License](http://opensource.org/licenses/MIT).

