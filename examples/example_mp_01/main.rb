$LOAD_PATH.unshift File.expand_path('../../../lib', __FILE__)
require 'open3'
require 'sabina'

DIM = 2
K = 2
LOOP_NUM = 100
xrange = [-2.2, 2.2]
yrange = [-2.2, 2.2]

training_data = Sabina::MultilayerPerceptron.load_csv('training_data.csv')

options = {
  :layers => [
    Sabina::Layer::MPInputLayer.new(DIM),
    Sabina::Layer::MPHiddenLayer.new(16),
    Sabina::Layer::MPHiddenLayer.new(8),
    Sabina::Layer::MPOutputLayer.new(K)
  ],
  :mini_batch_size => 10,
  :learning_rate => 0.01,
  :training_data => training_data,
}

mp = Sabina::MultilayerPerceptron.new(options)



##################################################
# Make a chart of trainig data.
##################################################
Open3.popen3('gnuplot') do |gp_in, gp_out, gp_err|
  output_file = "./mp_training_data.png"
  gp_in.puts "set colorsequence classic"
  gp_in.puts "set terminal png size 480, 450"
  gp_in.puts "set output '#{output_file}'"
  gp_in.puts "set label 2 center at screen 0.5,0.9 'Training Data' font 'Helvetica,22'"
  gp_in.puts "set tmargin 3.2"
  gp_in.puts "set xlabel 'x0'"
  gp_in.puts "set ylabel 'x1'"
  gp_in.puts "set size square"
  gp_in.puts xrange.tap { |f, t| break "set xrange [#{f}:#{t}]" }
  gp_in.puts yrange.tap { |f, t| break "set yrange [#{f}:#{t}]" }
  plot = "plot "

  K.times do |k|
    plot << "'-' notitle pt 1 ps 1 lc #{k+1},\\\n"
  end
  plot.gsub!(/,\\\n\z/, "\n")

  K.times do |k|
    label = Array.new(K) { |j| j == k ? 1 : 0 }
    training_data.each do |data|
      if label == data[:d]
        data[:x].tap { |x, y| plot << "#{x}, #{y}\n" }
      end
    end
    plot << "e\n"
  end

  gp_in.puts plot
  puts output_file
  gp_in.puts "set output"
  gp_in.puts "exit"
  gp_in.close
end


##################################################
# Make a GIF of the learning process.
##################################################
Open3.popen3('gnuplot') do |gp_in, gp_out, gp_err|
  output_file = "./mp_learning_process.gif"
  gp_in.puts "set colorsequence classic"
  gp_in.puts "set terminal gif animate delay 10 optimize size 880, 450"
  gp_in.puts "set output '#{output_file}'"
  gp_in.puts "set tmargin 3.2"
  gp_in.puts "set size square"
  log = []
  tmp_data = xrange.map { |v| (20*v).to_i }.tap { |f, t| break [*f..t] }
    .product yrange.map { |v| (20*v).to_i }.tap { |f, t| break [*f..t] }
  tmp_data.map! do |data|
    data.each_with_object(0.05).map(&:*)
  end
  x_mat = Matrix.columns( tmp_data )

  LOOP_NUM.times do |t|
    mp.learn
    log << mp.error(training_data)
    gp_in.puts "set multiplot layout 1, 2"

    ##################################################
    # Training Data
    ##################################################
    gp_in.puts "set label 2 center at screen 0.28,0.9 'Training Data' font 'Helvetica,22'"
    gp_in.puts xrange.tap { |f, t| break "set xrange [#{f}:#{t}]" }
    gp_in.puts yrange.tap { |f, t| break "set yrange [#{f}:#{t}]" }
    gp_in.puts "set xlabel 'x0'"
    gp_in.puts "set ylabel 'x1'"
    gp_in.puts "unset grid"
    plot = "plot "

    K.times do |k|
      plot << "'-' notitle pt 1 ps 1 lc #{k+1},\\\n"
    end
    K.times do |k|
      plot << "'-' notitle pt 7 ps 1 lc #{k+1},\\\n"
    end
    plot.gsub!(/,\\\n\z/, "\n")

    y_ary = mp.propagate_forward(x_mat).t.to_a
    K.times do |k|
      tmp_data.each_with_index do |(x, y), n|
        label = y_ary[n].index( y_ary[n].max )
        plot << "#{x}, #{y}\n" if label == k
      end
      plot << "e\n"
    end

    K.times do |k|
      label = Array.new(K) { |j| j == k ? 1 : 0 }
      training_data.each do |data|
        if label == data[:d]
          data[:x].tap { |x, y| plot << "#{x}, #{y}\n" }
        end
      end
      plot << "e\n"
    end
    gp_in.puts plot

    ##################################################
    # Training Error
    ##################################################
    gp_in.puts "set label 2 center at screen 0.79,0.9 'Training Error' font 'Helvetica,22'"
    gp_in.puts "set xrange [0:#{LOOP_NUM}]"
    gp_in.puts "set yrange [0:#{log.first + 10}]"
    gp_in.puts "set xlabel 'iteration number'"
    gp_in.puts "set ylabel 'training error'"
    gp_in.puts "set grid"
    plot = "plot "

    plot << "'-' notitle with lines lw 3 lt 1 lc 1,\\\n"
    plot.gsub!(/,\\\n\z/, "\n")
    log.each { |x, y| plot << "#{x}, #{y}\n" }
    plot << "e\n"
    gp_in.puts plot

    # progress bar
    puts " error : #{log.last}"
    puts " [#{("*"*((t.to_f / LOOP_NUM)*10).to_i).ljust(9, " ")}]"
    print "\e[2A"; STDOUT.flush;
  end

  gp_in.puts "set output"
  puts output_file
  gp_in.puts "exit"
  gp_in.close
end
