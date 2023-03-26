# reset sensor
echo 19 > /sys/class/gpio/export
echo out > /sys/class/gpio/gpio19/direction
echo 0 > /sys/class/gpio/gpio19/value
sleep 0.2
echo 1 > /sys/class/gpio/gpio19/value
echo 19 > /sys/class/gpio/unexport
echo 1 > /sys/class/vps/mipi_host0/param/snrclk_en
echo 24000000 > /sys/class/vps/mipi_host0/param/snrclk_freq
echo 1 > /sys/class/vps/mipi_host0/param/stop_check_instart
i2cdetect -y -r 1
