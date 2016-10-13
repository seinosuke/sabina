module Sabina
  class AutoEncoder < MultilayerPerceptron

    def self.load_csv(file_name)
      table = CSV.table(file_name)
      table.map do |data|
        x = data[0..-2]
        { :x => x, :d => x }
      end
    end

    # Check if `@layers` is valid.
    def check_layers
      super

      if @layers.size != 3
        raise "The number of layers must be three."
      end

      if @layers.first.size != @layers.last.size
        raise "The number of units of the input layer must be equal to that of the output layer."
      end
    end

    # Error function (a example of squared error)
    def error(test_data)
      d = Matrix.columns( test_data.map { |data| data[:d] } )
      y = propagate_forward(d)
      (d - y).to_a.flatten.inject(0.0) { |sum, v| sum + v**2 }
    end

    def next_input_data(input_data)
      x = Matrix.columns( input_data.map { |data| data[:x] } )
      propagate_forward(x)
      @Z[1].t.to_a.map do |z|
        { :x => z, :d => z }
      end
    end

    # Update the weights of this auto-encoder.
    def update
      super
      @layers[2].W = Marshal.load(Marshal.dump(@layers[1].W.t))
    end
  end
end
