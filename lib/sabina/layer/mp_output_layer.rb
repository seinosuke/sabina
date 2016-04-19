module Sabina::Layer
  class MPOutputLayer < BaseLayer

    # softmax function
    def activate(u_ary)
      sum = u_ary.inject(0.0) { |s, u| s + Math.exp(u) }
      u_ary.map do |u|
        Math.exp(u) / sum
      end
    end
  end
end
