require 'spec_helper'

describe Sabina::Configuration do
  let(:training_data) do
    Sabina::MultilayerPerceptron.load_csv("./spec/training_data.csv")
  end

  describe ".configure" do
    context "when a MultilayerPerceptron object is initialized with options" do
      it "should be overwritten" do
        Sabina.configure do |config|
          config.layers = [
            Sabina::Layer::MPInputLayer.new(2),
            Sabina::Layer::MPHiddenLayer.new(8),
            Sabina::Layer::MPOutputLayer.new(3)
          ]
          config.mini_batch_size = 30
          config.learning_rate = 0.03
          config.training_data = training_data
        end

        options = {
          :mini_batch_size => 20,
        }

        mp_01 = Sabina::MultilayerPerceptron.new
        mp_02 = Sabina::MultilayerPerceptron.new(options)

        expect(mp_01.mini_batch_size).to eq 30
        expect(mp_02.mini_batch_size).to eq 20
        expect(mp_01.learning_rate).to eq 0.03
        expect(mp_02.learning_rate).to eq 0.03
      end
    end
  end

  describe ".reset" do
    it "resets all options to their default values" do
      Sabina.configure do |config|
        config.layers = [
          Sabina::Layer::MPInputLayer.new(2),
          Sabina::Layer::MPHiddenLayer.new(8),
          Sabina::Layer::MPOutputLayer.new(3)
        ]
        config.mini_batch_size = 30
        config.learning_rate = 0.03
        config.training_data = training_data
      end

      expect do
        Sabina.reset
      end.to change{ Sabina.mini_batch_size }.from(30).to(10)
    end
  end
end
