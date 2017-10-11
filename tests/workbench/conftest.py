import pytest

import enaml
from enaml.workbench.api import Workbench
from enaml.qt.qt_application import QtApplication


with enaml.imports():
    from enaml.workbench.core.core_manifest import CoreManifest
    from psi.context.manifest import ContextManifest
    from psi.data.manifest import DataManifest
    from .helper_manifest import HelperManifest


@pytest.fixture(scope='session')
def app():
    return QtApplication()


@pytest.fixture
def workbench():
    workbench = Workbench()
    workbench.register(CoreManifest())
    workbench.register(ContextManifest())
    workbench.register(DataManifest())
    workbench.register(HelperManifest())

    context = workbench.get_plugin('psi.context')
    item = context.context_items['repetitions']
    item.rove = True
    for r in (20, 15, 10, 2):
        context.selectors['default'].add_setting({'repetitions': r})
    return workbench
