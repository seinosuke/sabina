module Sabina
  module Configuration

    OPTIONS_KEYS = [
      :layers,
      :mini_batch_size,
      :training_data,
      :learning_rate,
    ].freeze

    DEFAULTS = {
      :layers => [
        Sabina::Layer::MPInputLayer.new(1),
        Sabina::Layer::MPHiddenLayer.new(1),
        Sabina::Layer::MPOutputLayer.new(1)
      ],
      :mini_batch_size => 10,
      :training_data => [{:x => [0], :d => [0]}],
      :learning_rate => 0.01,
    }

    attr_accessor *OPTIONS_KEYS

    # This method is used for setting configuration options.
    def configure
      yield self
    end

    # Create a hash of options.
    def options
      Hash[*OPTIONS_KEYS.map{ |key| [key, send(key)] }.flatten(1)]
    end

    # Reset all options to their default values.
    def reset
      DEFAULTS.each do |option, default|
        self.send("#{option}=".to_sym, default)
      end
    end
  end
end
