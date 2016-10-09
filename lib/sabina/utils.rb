module Sabina
  module Utils
    def box_muller(s = 1.0)
      r_1 = rand
      r_2 = rand
      s * Math.sqrt(-2*Math.log(r_1)) * Math.cos(2*Math::PI*r_2)
    end

    module_function :box_muller
  end
end
