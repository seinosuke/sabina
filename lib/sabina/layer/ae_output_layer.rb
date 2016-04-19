module Sabina::Layer
  class AEOutputLayer < BaseLayer

    def initialize(size)
      super
      @f = ->(x){ x }
      @f_ = ->(_){ 1.0 }
    end
  end
end
