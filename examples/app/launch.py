def main():
    import logging
    logging.basicConfig(level='DEBUG')

    from psi.experiment.workbench import PSIWorkbench
    workbench = PSIWorkbench()

    io_manifest = 'io_manifest.IOManifest'
    controller_manifests = ['simple_experiment.ControllerManifest']
    workbench.register_core_plugins(io_manifest, controller_manifests)
    workbench.start_workspace('demo')


if __name__ == '__main__':
    main()
