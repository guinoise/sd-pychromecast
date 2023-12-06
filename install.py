import launch

mods= ["pychromecast", "Pillow"]

for m in mods:
    if not launch.is_installed(m):
        launch.run_pip("install {}".format(m), "requirements for sd-pychromecast")
