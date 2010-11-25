import sys

import dolo


from PyQt4 import QtCore, QtGui, uic

app = QtGui.QApplication(sys.argv)
       
[EqWidgetUi,EqWidgetBase] = uic.loadUiType("equation_widget.ui")

class MainWindow(QtGui.QMainWindow):
#    def accept(self):h
#        print 'hello'
#        self.eql.add_equation()
    current_file = None

    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        
        # Set up the user interface from Designer.
        self.ui = uic.loadUi("modfile_editor.ui")

#        self.ew = uic.loadUi("equation_widget.ui")

        # Connect up the buttons.
        self.connect(self.ui.pushButton, QtCore.SIGNAL("clicked()"), self, QtCore.SLOT("add_widget()"))
        self.connect(self.ui.pushButton_2, QtCore.SIGNAL("clicked()"), self, QtCore.SLOT("check()"))
        self.connect(self.ui.actionOpen, QtCore.SIGNAL("activated()"), self, QtCore.SLOT("open()"))
        self.connect(self.ui.actionSave, QtCore.SIGNAL("activated()"), self, QtCore.SLOT("save_as()"))
        self.add_widget()
        self.ui.show()

    @QtCore.pyqtSlot()
    def open(self):
        filename = QtGui.QFileDialog.getOpenFileName()
        filename = str(filename)
        try:
            f = file( filename )
            txt = f.read()
        finally:
            f.close()
        try:
            model = dolo.dynare_import(filename)
            model.check(verbose=True)
            n = len(model.equations)
            q = len(self.widgets)
            if n > q:
                for i in range(n-q):
                    self.add_widget()
            elif n < q:
                for i in range(q-n):
                    self.delete_widget(0)
            for n,eq in enumerate(model.equations):
                tt = str(eq)
                tt = tt.replace('**','^')
                tt = tt.replace('==','=')
                self.widgets[n].set_text(tt)
            self.ui.lineEdit_var.setText( str.join(' ',[ str(v) for v in model.variables ]) )
            self.ui.lineEdit_parameters.setText( str.join(' ',[ str(v) for v in model.parameters ]) )
            self.ui.lineEdit_shocks.setText(  str.join(' ',[ str(v) for v in model.shocks ]) )

        except Exception as e:
            print 'Import failed.'
            print e


    @QtCore.pyqtSlot()            
    def add_widget(self):
        ew = CustomWidget()
        ew.father = self
#        ew = EquationWidget()
#        ew.populate()
        n = self.ui.verticalLayout_3.count()
        self.ui.verticalLayout_3.addWidget(ew)
        self.widgets.append(ew)
        ew.setup( n )
#        self.equations_widgets.append(ew)
        self.connect(ew, QtCore.SIGNAL("hi(int)"), self, QtCore.SLOT("delete_widget(int)"))

    i = 0
    widgets = []
    
    @QtCore.pyqtSlot(int)        
    def delete_widget(self,n):
        widget = self.widgets[n]
        self.ui.verticalLayout_3.removeWidget(widget)
        self.widgets.remove(widget)
        widget.hide()
        del widget
        for i,w in enumerate(self.widgets):
            w.set_number(i)

    @QtCore.pyqtSlot()
    def save_as(self):
        filename = QtGui.QFileDialog.getSaveFileName()
        l = [ el.get_text() for el in self.widgets]
        l = [ e for e in l if e]
        content = str.join(';\n',l)
        try:
            f = file(filename,'w')
            f.write(content)
        except Exception as e:
            print e
        finally:
            f.close()


    @QtCore.pyqtSlot()
    def check(self):
        var_string = self.ui.lineEdit_var.text()
        shocks_string = self.ui.lineEdit_shocks.text()
        parameters_string = self.ui.lineEdit_parameters.text()
        l = [ el.get_text() for el in self.widgets]
        l = [ e for e in l if e]
        content = str.join(';\n',l)
        if len(l)>0:
            content += ';'
        simple_mod = '''
var {vars};
varexo {varexo};
parameters {parms};
model;
{equations}
end;
        '''.format(vars = var_string,varexo = shocks_string,
        parms=parameters_string,equations = content)

        from dolo.misc.interactive import parse_dynare_text
        model = parse_dynare_text(simple_mod)
        try:
            model.check(verbose=True)
        except Exception as e:
            print 'Model is not valid'

        info = model.info
        print model.fname
        info['name'] = model.fname
        txt ='''
Model check {name}:
    Number of variables :  {n_variables}
    Number of equations :  {n_equations}
    Number of shocks :     {n_shocks}
        '''.format(**info)
        self.ui.textEdit.setText(txt)
        

    def declared_symbols(self):
        vars = [dolo.Variable(e,0) for e in str.split(str(self.ui.lineEdit_var.text()),' ' ) ]
        varexos = [dolo.Shock(e,0) for e in str.split(str(self.ui.lineEdit_shocks.text()),' ' ) ]
        parms = [dolo.Parameter(e) for e in str.split(str(self.ui.lineEdit_parameters.text()),' ') ]
        context = {}
        for s in vars + varexos + parms:
            print s
            context[s.name] = s
        return context
        
import sympy

class CustomWidget(QtGui.QWidget):
    def do_nothing():
        print 'do'
    def setup(self,n):

        self.n = n
        self.ui = EqWidgetUi()
        self.ui.setupUi(self)
        self.ui.frame.setVisible(False)
        self.connect(self.ui.toolButton, QtCore.SIGNAL("clicked()"), self, QtCore.SLOT("specialSignal()"))
        self.connect(self.ui.pushButton, QtCore.SIGNAL("clicked()"), self, QtCore.SLOT("check()"))
        self.connect(self.ui.pushButton_2, QtCore.SIGNAL("clicked()"), self, QtCore.SLOT("def_parameter()"))
        self.connect(self.ui.pushButton_3, QtCore.SIGNAL("clicked()"), self, QtCore.SLOT("def_variable()"))
        self.connect(self.ui.pushButton_4, QtCore.SIGNAL("clicked()"), self, QtCore.SLOT("def_shock()"))

        self.locals = {}

    @QtCore.pyqtSlot()
    def def_parameter(self):
        t = self.undefined
        tt = str( self.father.ui.lineEdit_parameters.text() )
        self.father.ui.lineEdit_parameters.setText( (tt + ' ' + t).strip(' ') )
        self.check()

    @QtCore.pyqtSlot()
    def def_shock(self):
        t = self.undefined
        tt = str( self.father.ui.lineEdit_shocks.text() )
        self.father.ui.lineEdit_shocks.setText(( tt + ' ' + t).strip(' '))
        self.check()
        
    @QtCore.pyqtSlot()
    def def_variable(self):
        t = self.undefined
        tt = str( self.father.ui.lineEdit_var.text() )
        self.father.ui.lineEdit_var.setText( ( tt + ' ' + t) .strip(' ') )
        self.check()
                
    @QtCore.pyqtSlot()        
    def specialSignal(self):
        self.emit(QtCore.SIGNAL('hi(int)'),self.n)

    @QtCore.pyqtSlot()        
    def check(self):
        txt = self.get_text()
        d = self.father.declared_symbols()

        try:
            txt = txt.replace('^','**')
            if '=' in txt:
                lhs,rhs = txt.split('=')
                txt = lhs + '==' + rhs
            eval(txt,d)
            self.undefined = None
            msg = 'Hiding'
            self.ui.label.setText(msg)
            self.ui.frame.setVisible(False)
        except NameError as e:
            import re
            missing_name = re.compile("name \'(.*)\'").findall(str(e))[0]
            msg = "Symbol ''{0}'' is not defined. Define it as :".format(missing_name)
            self.undefined = missing_name
            self.ui.label.setText(msg)
            self.ui.frame.setVisible(True)

        
    def set_number(self,n):
        self.n = n
        
    def get_text(self):
        txt = self.ui.lineEdit.text()
        if txt == '':
            return None
        else:
            return str(txt)

    def set_text(self,tt):
        self.ui.lineEdit.setText(tt)



window = MainWindow()
sys.exit(app.exec_())
