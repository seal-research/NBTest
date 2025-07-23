import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';
import { ICommandPalette, ToolbarButton } from '@jupyterlab/apputils';
import { INotebookTracker, NotebookPanel } from '@jupyterlab/notebook';
import { IRenderMimeRegistry } from '@jupyterlab/rendermime';
import { Signal } from '@lumino/signaling';
import { Widget } from '@lumino/widgets';
import { CodeCellModel } from '@jupyterlab/cells';

// Signal for updating the status display of the ENV variable
class ToggleSignal {
  private _stateChanged = new Signal<this, string>(this);

  get stateChanged() {
    return this._stateChanged;
  }

  emitState(value: string) {
    this._stateChanged.emit(value);
  }
}

const toggleSignal = new ToggleSignal();
let status = 0; // Track status locally for the ENV variable

// Define constants for the metadata key and assertion prefix
const METADATA_KEY = 'nbtest_hidden_asserts';
const ASSERT_PREFIX = 'nbtest.assert_';

/**
 * The main extension plugin.
 */
const plugin: JupyterFrontEndPlugin<void> = {
  id: 'nbtest_lab_extension:plugin',
  autoStart: true,
  requires: [ICommandPalette, INotebookTracker, IRenderMimeRegistry],
  activate: (
    app: JupyterFrontEnd,
    palette: ICommandPalette,
    tracker: INotebookTracker
  ) => {
    const { commands } = app;
    const toggleEnvCommand = 'nbtest:toggle-asserts-env';
    const toggleVisibilityCommand = 'nbtest:toggle-visibility';

    // COMMAND 1: Toggle NBTEST_RUN_ASSERTS environment variable
    commands.addCommand(toggleEnvCommand, {
      label: 'Toggle NBTEST_RUN_ASSERTS Env Var',
      execute: async () => {
        const currentNotebook = tracker.currentWidget;
        if (!currentNotebook) {
          return;
        }
        const session = currentNotebook.sessionContext.session;
        if (!session || !session.kernel) {
          return;
        }

        const code = `
import os
os.environ["NBTEST_RUN_ASSERTS"] = "1" if os.environ.get("NBTEST_RUN_ASSERTS", "0") != "1" else "0"
print(os.environ["NBTEST_RUN_ASSERTS"])
        `;
        const future = session.kernel.requestExecute({ code });
        future.onIOPub = msg => {
          if (msg.header.msg_type === 'stream') {
            const newStatusValue = (msg.content as any).text.trim();
            status = newStatusValue === '1' ? 1 : 0;
            toggleSignal.emitState(status === 1 ? 'ON' : 'OFF');
          }
        };
        await future.done;
      }
    });

    // COMMAND 2: Completely hide or show nbtest.assert_* lines
    commands.addCommand(toggleVisibilityCommand, {
      label: 'Hide/Show NBTest Assertions',
      execute: () => {
        const panel = tracker.currentWidget;
        if (!panel) {
          return;
        }

        const notebookModel = panel.content.model;
        if (!notebookModel) {
          return;
        }

        const cells = notebookModel.cells;
        let shouldHide = true;

        // Check if any cell has hidden assertions in its metadata
        for (let i = 0; i < cells.length; i++) {
          const cell = cells.get(i);
          if (cell.type !== 'code') {
            continue;
          }
          if (cell.getMetadata(METADATA_KEY) !== undefined) {
            shouldHide = false;
            break;
          }
        }

        // Apply the determined action
        for (let i = 0; i < cells.length; i++) {
          const cellModel = cells.get(i);
          if (cellModel.type !== 'code') {
            continue;
          }

          if (shouldHide) {
            const sourceLines = cellModel.sharedModel.getSource().split('\n');
            const visibleLines: string[] = [];
            const hiddenLines: string[] = [];

            sourceLines.forEach((line: string) => {
              if (line.trim().startsWith(ASSERT_PREFIX)) {
                hiddenLines.push(line);
              } else {
                visibleLines.push(line);
              }
            });

            if (hiddenLines.length > 0) {
              cellModel.sharedModel.setSource(visibleLines.join('\n'));
              cellModel.setMetadata(METADATA_KEY, hiddenLines);
            }
          } else {
            const hiddenLines = cellModel.getMetadata(METADATA_KEY) as string[];
            if (hiddenLines) {
              const visibleLines = cellModel.sharedModel.getSource();
              const separator = visibleLines.trim().length > 0 ? '\n' : '';

              cellModel.sharedModel.setSource(
                visibleLines + separator + hiddenLines.join('\n')
              );
              cellModel.deleteMetadata(METADATA_KEY);
            }
          }
        }
      }
    });

    // Add commands to the palette
    palette.addItem({ command: toggleEnvCommand, category: 'NBTest' });
    palette.addItem({ command: toggleVisibilityCommand, category: 'NBTest' });

    // Add buttons and functionality to any new notebook
    tracker.widgetAdded.connect((sender, panel: NotebookPanel) => {
      const envButton = new ToolbarButton({
        label: 'Toggle Assertions',
        tooltip: 'Toggle NBTEST_RUN_ASSERTS Environment Variable',
        onClick: () => commands.execute(toggleEnvCommand)
      });

      const statusDisplay = new Widget();
      statusDisplay.node.textContent = 'NBTest Status: OFF';
      statusDisplay.node.style.marginLeft = '4px';
      statusDisplay.node.style.marginRight = '8px';

      toggleSignal.stateChanged.connect((_, newStatus) => {
        statusDisplay.node.textContent = `NBTest Status: ${newStatus}`;
      });

      const visibilityButton = new ToolbarButton({
        label: 'Hide/Show Assertions',
        tooltip: 'Completely hide or show nbtest assertions',
        onClick: () => commands.execute(toggleVisibilityCommand)
      });

      panel.toolbar.addItem('toggleAssertsEnv', envButton);
      panel.toolbar.addItem('assertsStatus', statusDisplay);
      panel.toolbar.addItem('toggleVisibility', visibilityButton);

      const highlightAssertCells = () => {
        panel.content.widgets.forEach(cell => {
          const model = cell.model;
          const node = cell.node;
          let hasAssertions = false;

          if (model instanceof CodeCellModel) {
            const source = model.sharedModel.getSource();
            const hasVisibleAssertions = /nbtest\.assert_\w+/.test(source);
            const hasHiddenAssertions =
              model.getMetadata(METADATA_KEY) !== undefined;
            hasAssertions = hasVisibleAssertions || hasHiddenAssertions;
          }

          if (hasAssertions) {
            node.style.borderLeft = '4px solid #f39c12';
            node.style.backgroundColor = 'rgba(243, 156, 18, 0.07)';
          } else {
            node.style.borderLeft = '';
            node.style.backgroundColor = '';
          }
        });
      };

      // Run highlighting once the panel is ready.
      panel.revealed.then(() => {
        highlightAssertCells();
      });

      // Re-run highlighting efficiently on any content change.
      if (panel.content.model) {
        panel.content.model.contentChanged.connect(() => {
          highlightAssertCells();
        });
      }
    });
  }
};

export default plugin;
