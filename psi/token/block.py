import copy

from atom.api import Unicode, Callable, Typed, Property

from enaml.core.api import Declarative, d_

from psi.context.api import Parameter


class Block(Declarative):

    name = d_(Unicode())
    label = d_(Unicode())
    factory = d_(Callable())
    context_name_map = Typed(dict)

    blocks = Property()
    parameters = Property()

    def _get_blocks(self):
        return self.get_children(Block)

    def _get_parameters(self):
        return self.get_children(Parameter)

    def get_children(self, child_type):
        return [c for c in self.children if isinstance(c, child_type)]

    def configure_context_items(self, output_name, output_label, scope):
        self.context_name_map = {}
        for item in self.parameters:
            old_name = item.name
            item.name = '{}_{}_{}'.format(output_name, self.name, item.name)
            item.label = '{} {}'.format(self.label, item.label)
            item.compact_label = '{} {} {}'.format(output_label, self.label,
                                                   item.compact_label)
            item.group = output_name
            item.scope = scope
            self.context_name_map[old_name] = item.name

        for block in self.blocks:
            block.configure_context_items(output_name, output_label, scope)

    def get_context_items(self):
        items = self.get_children(Parameter)
        for block in self.get_children(Block):
            items.extend(block.get_context_items())
        return items

    def get_context_names(self):
        return [item.name for item in self.get_context_items()] 

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.name = self.name
        result.label = self.label
        result.factory = self.factory
        for c in self.children:
            result.children.append(copy.copy(c))
        return result

    def initialize_generator(self, workbench, **context):
        inputs = [b.initialize_generator(workbench, **context) \
                  for b in self.blocks]
        context_plugin = workbench.get_plugin('psi.context')
        for generator_name, context_name in self.context_name_map.items():
            context[generator_name] = context_plugin.get_value(context_name)
        generator = self.factory(inputs=inputs, **context)
        generator.next()
        return generator
