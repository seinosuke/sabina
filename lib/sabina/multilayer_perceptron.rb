module Sabina
  class MultilayerPerceptron
    attr_accessor *Configuration::OPTIONS_KEYS
    LAMBDA = 0.001
    MU = 0.5

    def self.load_csv(file_name)
      table = CSV.table(file_name)
      k = table[:label].max + 1
      table.map do |data|
        x = (data.size-1).times.map { |d| data["x#{d}".to_sym] }
        d = Array.new(k) { |i| i == data[:label] ? 1 : 0 }
        { :x => x, :d => d }
      end
    end

    def initialize(options = {})
      load_config(options)
      check_layers
      @L = @layers.size - 1
      @K = @layers.last.size
      @delta_W_prev = []
      @delta_b_prev = []

      @layers[0].J = @layers[0].size
      (1..@L).each do |j|
        @layers[j].I = @layers[j-1].size
        @layers[j].J = @layers[j].size
        @layers[j].init_weight
        @delta_W_prev[j] = Marshal.load(Marshal.dump(@layers[j].W * 0.0))
        @delta_b_prev[j] = Marshal.load(Marshal.dump(@layers[j].b * 0.0))
      end

      @mini_batches =
        @training_data.shuffle
        .each_slice(@mini_batch_size).to_a
    end

    # Load the configuration.
    def load_config(options = {})
      merged_options = Sabina.options.merge(options)
      Configuration::OPTIONS_KEYS.each do |key|
        send("#{key}=".to_sym, merged_options[key])
      end
    end

    # Check if `@layers` is valid.
    def check_layers
      if layers.size < 3
        raise "The number of layers size must be more than three."
      end
    end

    # Error function (a example of cross entropy)
    def error(test_data)
      x = Matrix.columns( test_data.map { |data| data[:x] } )
      y = propagate_forward(x)
      test_data.each_with_index.inject(0.0) do |mn, (data, n)|
        mn + data[:d].each_with_index.inject(0.0) do |mk, (d, k)|
          mk - d * Math.log(y[k, n])
        end
      end
    end

    # A learning step consists of 
    # a forward propagation, a backward propagation
    # and updating the weights of this multi-layer perceptron.
    def learn
      @mini_batches.each do |mini_batch|
        @X = Matrix.columns( mini_batch.map { |data| data[:x] } ) # (Dim, N)
        @D = Matrix.columns( mini_batch.map { |data| data[:d] } ) # (Dim, N)

        propagate_forward(@X)
        propagate_backward
        update
      end
    end

    # Input values are propagated forward.
    def propagate_forward(x_mat)
      @X = x_mat
      @N = @X.column_size
      @Z, @U = [], []

      # l = 0
      @Z[0] = @X

      # l = 1..L
      ones = Matrix[Array.new(@N) { 1.0 }]
      (1..@L).each do |l|
        @U[l] = @layers[l].W*@Z[l-1] + @layers[l].b*ones
        @Z[l] = Matrix.columns( @U[l].t.to_a.map { |u| @layers[l].activate(u) } )
      end

      # Oputput (K, N)
      @Y = @Z[@L]
    end

    # Training errors are propagated backwards.
    def propagate_backward
      @Delta = []

      # l = L
      @Delta[@L] = @Y - @D

      # l = (L-1)..1
      [*(1...@L)].reverse.each do |l|
        f_u = Matrix.columns( @U[l].t.to_a.map { |u| @layers[l].activate_(u) } )
        w_d = @layers[l+1].W.t * @Delta[l+1]
        @Delta[l] = @layers[l].J.times.map do |j|
          @N.times.map do |n|
            f_u[j, n] * w_d[j, n]
          end
        end.tap { |ary| break Matrix[*ary] }
      end
    end

    # Update the weights of this multi-layer perceptron.
    def update
      ones = Matrix.columns( [Array.new(@N) { 1.0 }] )
      (1..@L).each do |l|
        delta_W = ( MU*@delta_W_prev[l] ) -
          @learning_rate*( (@Delta[l] * @Z[l-1].t / @N) + LAMBDA*@layers[l].W )
        delta_b = ( MU*@delta_b_prev[l] ) -
          @learning_rate*( @Delta[l] * ones )

        @delta_W_prev[l] = Marshal.load(Marshal.dump(delta_W))
        @delta_b_prev[l] = Marshal.load(Marshal.dump(delta_b))

        @layers[l].W = @layers[l].W + delta_W
        @layers[l].b = @layers[l].b + delta_b
      end
    end
  end
end
