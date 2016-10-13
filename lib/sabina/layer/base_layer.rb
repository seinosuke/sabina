module Sabina::Layer
  class BaseLayer
    attr_reader :size
    attr_accessor :b, :W, :I, :J

    def initialize(size)
      @size = size
      @f = ->(x){ 1.0 / (1.0 + Math.exp(-x)) }
      @f_ = ->(x){ @f[x]*(1.0 - @f[x]) }
    end

    # Initialize the weights of this layer.
    def init_weight
      # (J, 1)
      @b = Matrix.columns([Array.new(@size) { 0.0 }])

      # (J, I)
      @W = Array.new(@J) do
        Array.new(@I) { Sabina::Utils.box_muller }
      end.tap { |ary| break Matrix[*ary] }
    end

    # An activation function
    def activate(u_ary)
      u_ary.map { |u| @f[u] }
    end

    # Differentiation of activation function
    def activate_(u_ary)
      u_ary.map { |u| @f_[u] }
    end
  end
end
