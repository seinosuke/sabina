require 'spec_helper'

describe Sabina::MultilayerPerceptron do
  let(:training_data) do
    Sabina::MultilayerPerceptron.load_csv("./spec/training_data.csv")
  end

  describe ".load_csv" do
    it "returns a array of training data" do
      expect(training_data.class).to eq Array
    end
  end

  describe "#check_layers" do
    context "when @layers is valid" do
      it "quit successfully" do
        options = {
          :layers => [
            Sabina::Layer::MPInputLayer.new(2),
            Sabina::Layer::MPHiddenLayer.new(8),
            Sabina::Layer::MPOutputLayer.new(3)
          ],
          :mini_batch_size => 10,
          :learning_rate => 0.01,
          :training_data => training_data,
        }

        expect( Sabina::MultilayerPerceptron.new(options).class )
          .to eq Sabina::MultilayerPerceptron
      end
    end

    context "when the number of @layers size is less than three" do
      it "raise RuntimeError" do
        options = {
          :layers => [
            Sabina::Layer::MPInputLayer.new(2),
            Sabina::Layer::MPOutputLayer.new(3)
          ],
          :mini_batch_size => 10,
          :learning_rate => 0.01,
          :training_data => training_data,
        }

        expect{ Sabina::MultilayerPerceptron.new(options) }
          .to raise_error RuntimeError
      end
    end
  end
end
