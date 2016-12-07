namespace :ssh do
  task :carnd do
    sh 'ssh carnd@54.200.226.53'
  end

  task :carnd2 do
    `chmod 400 rakelib/carkeys.pem`
    sh 'ssh -i rakelib/carkeys.pem ubuntu@54.145.117.121'
  end
end

namespace :scp do
  namespace :carnd do
    task :up, [:src, :dest] do |t, args|
      host = 'carnd@54.200.226.53'
      puts "uploading #{args[:src]} to #{host}:#{args[:dest]}"
      sh "scp -rp #{args[:src]} #{host}:#{args[:dest]}"
    end

    task :down, [:src, :dest] do |t, args|
      args.with_defaults(dest: '~')
      host = 'carnd@54.200.226.53'
      puts "downloading #{host}:#{args[:src]} to #{args[:dest]}"
      sh "scp -rp #{host}:#{args[:src]} #{args[:dest]}"
    end
  end

  namespace :carnd2 do
    task :up, [:src, :dest] do |t, args|
      host = 'carnd@54.145.117.121'
      puts "uploading #{args[:src]} to #{host}:#{args[:dest]}"
      sh "scp -rp #{args[:src]} #{host}:#{args[:dest]}"
    end

    task :down, [:src, :dest] do |t, args|
      args.with_defaults(dest: '~')
      host = 'carnd@54.145.117.121'
      puts "downloading #{host}:#{args[:src]} to #{args[:dest]}"
      sh "scp -rp #{host}:#{args[:src]} #{args[:dest]}"
    end
  end

end