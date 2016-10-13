$LOAD_PATH.unshift File.expand_path('../../../lib', __FILE__)
require 'open3'
require 'sabina'

DIM = 2
EPOCH = 100
xrange = [-2.2, 2.2]
yrange = [-2.2, 2.2]

original_data = Sabina::AutoEncoder.load_csv('training_data.csv')

options = {
  :layers => [
    Sabina::Layer::AEInputLayer.new(DIM),
    Sabina::Layer::AEHiddenLayer.new(20),
    Sabina::Layer::AEOutputLayer.new(DIM)
  ],
  :mini_batch_size => 10,
  :learning_rate => 0.0002,
  :training_data => original_data,
}

# Use a sparse autoencoder
# because the number of input units is less than that of hidden units.
sae = Sabina::SparseAutoEncoder.new(options)



##################################################
# Make a chart of original data.
##################################################
Open3.popen3('gnuplot') do |gp_in, gp_out, gp_err|
  output_file = "./ae_original_data.png"
  gp_in.puts "set colorsequence classic"
  gp_in.puts "set terminal png size 480, 450"
  gp_in.puts "set output '#{output_file}'"
  gp_in.puts "set label 2 center at screen 0.5,0.9 'Original Data' font 'Helvetica,22'"
  gp_in.puts "set tmargin 3.2"
  gp_in.puts "set xlabel 'x0'"
  gp_in.puts "set ylabel 'x1'"
  gp_in.puts "set size square"
  gp_in.puts xrange.tap { |f, t| break "set xrange [#{f}:#{t}]" }
  gp_in.puts yrange.tap { |f, t| break "set yrange [#{f}:#{t}]" }
  plot = "plot "

  plot << "'-' notitle pt 1 ps 1 lc 1,\\\n"
  plot.gsub!(/,\\\n\z/, "\n")

  original_data.each do |data|
    data[:x].tap { |x, y| plot << "#{x}, #{y}\n" }
  end
  plot << "e\n"

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
  output_file = "./ae_learning_process.gif"
  gp_in.puts "set colorsequence classic"
  gp_in.puts "set terminal gif animate delay 10 optimize size 880, 450"
  gp_in.puts "set output '#{output_file}'"
  gp_in.puts "set tmargin 3.2"
  gp_in.puts "set size square"
  gp_in.puts xrange.tap { |f, t| break "set xrange [#{f}:#{t}]" }
  gp_in.puts yrange.tap { |f, t| break "set yrange [#{f}:#{t}]" }
  gp_in.puts "set xlabel 'x0'"
  gp_in.puts "set ylabel 'x1'"
  log = []
  x_mat = Matrix.columns( original_data.map { |data| data[:x] } )

  EPOCH.times do |t|
    sae.learn
    log << sae.error(original_data)
    gp_in.puts "set multiplot layout 1, 2"

    ##################################################
    # Original Data
    ##################################################
    gp_in.puts "set label 2 center at screen 0.28,0.9 'Original Data' font 'Helvetica,22'"
    plot = "plot "
    plot << "'-' notitle pt 1 ps 1 lc 1,\\\n"
    plot.gsub!(/,\\\n\z/, "\n")

    original_data.each do |data|
      data[:x].tap { |x, y| plot << "#{x}, #{y}\n" }
    end
    plot << "e\n"
    gp_in.puts plot

    ##################################################
    # Decoded Data
    ##################################################
    gp_in.puts "set label 2 center at screen 0.79,0.9 'Decoded Data' font 'Helvetica,22'"
    plot = "plot "
    plot << "'-' notitle pt 1 ps 1 lc 1,\\\n"
    plot.gsub!(/,\\\n\z/, "\n")

    decoded_data = sae.propagate_forward(x_mat).t.to_a
    decoded_data.each do |data|
      data.tap { |x, y| plot << "#{x}, #{y}\n" }
    end
    plot << "e\n"
    gp_in.puts plot

    # progress bar
    puts " error : #{log.last}"
    puts " [#{("*"*((t.to_f / EPOCH)*10).to_i).ljust(9, " ")}]"
    print "\e[2A"; STDOUT.flush;
  end

  gp_in.puts "set output"
  puts output_file
  gp_in.puts "exit"
  gp_in.close
end
