module Sabina
  class SparseAutoEncoder < AutoEncoder
    BETA = 0.1

    def initialize(options = {})
      super

      @RHO_const = Matrix[Array.new(@layers[1].J) { 0.05 }]
      @RHO_prev = Matrix[Array.new(@layers[1].J) { 1.0 }]
    end

    # Errors are propagated backwards.
    def propagate_backward
      calc_rho
      @Delta = []

      # l = L
      @Delta[@L] = @Y - @D

      # l = 1
      l = 1
      f_u = Matrix.columns( @U[l].t.to_a.map { |u| @layers[l].activate_(u) } )
      w_d = @layers[l+1].W.t * @Delta[l+1]
      sps = @layers[l].J.times.map do |j|
        ((1.0 - @RHO_const[0, j]) / (1.0 - @RHO[0, j])) -
        (@RHO_const[0, j] / @RHO[0, j])
      end.tap do |ary|
        ary = ary.map { |v| v > 1e10 ? 1e10 : v }
        break Matrix.columns(Array.new(@N) { ary })
      end
      w_d_s = w_d + BETA*sps

      @Delta[l] = @layers[l].J.times.map do |j|
        @N.times.map do |n|
          f_u[j, n] * w_d_s[j, n]
        end
      end.tap { |ary| break Matrix[*ary] }
    end

    # Calculate average activities
    def calc_rho
      @RHO = @Z[1].to_a.map do |z_ary|
        z_ary.inject(0.0, :+)
      end.tap { |ary| break Matrix[ary] / @N }

      @RHO = 0.9*@RHO_prev + 0.1*@RHO
    end
  end
end
