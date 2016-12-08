# CARND_UP = '54.200.226.53'
CARND_UP = '54.191.69.58'
INSTANCE_ID = 'i-516ff7c4'

namespace :carnd do
  task :ssh do
    sh "ssh carnd@#{CARND_UP}"
  end

  task :start, [:instance_id] do |t, args|
    args.with_defaults(instance_id: INSTANCE_ID)
    instance_id = args[:instance_id]
    sh "aws ec2 start-instances --instance-ids \"#{instance_id}\""
  end

  task :stop, [:instance_id] do |t, args|
    args.with_defaults(instance_id: INSTANCE_ID)
    instance_id = args[:instance_id]
    sh "aws ec2 stop-instances --instance-ids \"#{instance_id}\""
  end

  namespace :scp do
    task :up, [:src, :dest] do |t, args|
      args.with_defaults(dest: '~')
      host = "carnd@#{CARND_UP}"
      puts "uploading #{args[:src]} to #{host}:#{args[:dest]}"
      sh "rsync -avz --exclude '*.zip' --exclude '*.pickle' --exclude '*.p' zimpy #{host}:#{args[:dest]}"
      sh "rsync -avz --exclude '*.zip' --exclude '*.pickle' --exclude '*.p' networks #{host}:#{args[:dest]}"
      unless args[:src].nil?
        sh "rsync -avz --exclude '*.zip' --exclude '*.pickle' --exclude '*.p' #{args[:src]} #{host}:#{args[:dest]}"
      end
    end

    task :down, [:src, :dest] do |t, args|
      args.with_defaults(dest: '~')
      host = "carnd@#{CARND_UP}"
      puts "downloading #{host}:#{args[:src]} to #{args[:dest]}"
      sh "scp -rp #{host}:#{args[:src]} #{args[:dest]}"
    end
  end
end