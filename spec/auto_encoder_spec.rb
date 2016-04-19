require 'spec_helper'

describe Sabina::AutoEncoder do
  let(:original_data) do
    Sabina::AutoEncoder.load_csv("./spec/training_data.csv")
  end

  describe ".load_csv" do
    it "returns a array of original data" do
      expect(original_data.class).to eq Array
    end
  end

  describe "#check_layers" do
    context "when @layers is valid" do
      it "quit successfully" do
        options = {
          :layers => [
            Sabina::Layer::AEInputLayer.new(2),
            Sabina::Layer::AEHiddenLayer.new(1),
            Sabina::Layer::AEOutputLayer.new(2)
          ],
          :mini_batch_size => 10,
          :learning_rate => 0.01,
          :training_data => original_data,
        }

        expect( Sabina::AutoEncoder.new(options).class )
          .to eq Sabina::AutoEncoder
      end
    end

    context "when the number of layers is not three" do
      it "raise RuntimeError" do
        options = {
          :layers => [
            Sabina::Layer::AEInputLayer.new(2),
            Sabina::Layer::AEHiddenLayer.new(1),
            Sabina::Layer::AEHiddenLayer.new(1),
            Sabina::Layer::AEOutputLayer.new(2)
          ],
          :mini_batch_size => 10,
          :learning_rate => 0.01,
          :training_data => original_data,
        }

        expect{ Sabina::AutoEncoder.new(options) }
          .to raise_error RuntimeError
      end
    end

    context "when the number of units of the input layer is not equal to that of the output layer" do
      it "raise RuntimeError" do
        options = {
          :layers => [
            Sabina::Layer::AEInputLayer.new(2),
            Sabina::Layer::AEHiddenLayer.new(1),
            Sabina::Layer::AEOutputLayer.new(3)
          ],
          :mini_batch_size => 10,
          :learning_rate => 0.01,
          :training_data => original_data,
        }

        expect{ Sabina::AutoEncoder.new(options) }
          .to raise_error RuntimeError
      end
    end
  end
end
