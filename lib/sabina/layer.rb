module Sabina
  module Layer
    autoload :MPInputLayer, "sabina/layer/mp_input_layer"
    autoload :MPHiddenLayer, "sabina/layer/mp_hidden_layer"
    autoload :MPOutputLayer, "sabina/layer/mp_output_layer"

    autoload :AEInputLayer, "sabina/layer/ae_input_layer"
    autoload :AEHiddenLayer, "sabina/layer/ae_hidden_layer"
    autoload :AEOutputLayer, "sabina/layer/ae_output_layer"
  end
end
